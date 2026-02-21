import os
import time
import logging
from typing import List
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydub import AudioSegment
import numpy as np
from supabase import create_client
from dotenv import load_dotenv
import uuid
import onnxruntime as ort
import librosa

# ---------------- ENV ----------------
load_dotenv()

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enroll_service")

# ---------------- ONNX MODEL LOAD ----------------
logger.info("Loading ONNX ECAPA model...")

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 2
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

onnx_session = ort.InferenceSession(
    "ecapa_embedding.onnx",
    sess_options=sess_options,
    providers=["CPUExecutionProvider"]
)

onnx_input_name = onnx_session.get_inputs()[0].name

logger.info("ONNX model loaded successfully")

# ---------------- ENV CONFIG ----------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

VECTOR_DIM = int(os.getenv("VECTOR_DIM", "192"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "ecapa_tdnn_v1")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE env vars missing")

# ---------------- SUPABASE ----------------
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# temp audio dir
TEMP_AUDIO_DIR = "audio_tmp"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# ---------------- FASTAPI ----------------
app = FastAPI(title="Voice Biometrics Enrollment Service")

bearer_scheme = HTTPBearer()

def get_user_from_session(session_bearer: str) -> dict:
    """
    Validate the user's session token with Supabase Auth and return the user object.
    Sends both Authorization and apikey headers (required by Supabase).
    """
    if not session_bearer:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    anon_key = os.getenv("SUPABASE_ANON_KEY")
    if not anon_key:
        logger.error("Missing SUPABASE_ANON_KEY in env")
        raise HTTPException(status_code=500, detail="Server misconfiguration")

    headers = {
        "Authorization": session_bearer,
        "apikey": anon_key,
    }
    resp = requests.get(f"{SUPABASE_URL}/auth/v1/user", headers=headers)
    logger.info("Supabase /auth/v1/user status=%s body=%s", resp.status_code, resp.text)
    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid session token")
    return resp.json()


def convert_to_wav(in_path: str, out_path: str, target_sr: int = 16000):
    """
    Convert input audio file to a WAV file using pydub (ffmpeg required).
    Exports 16-bit PCM WAV at target_sr sample rate.
    """
    audio = AudioSegment.from_file(in_path)
    audio = audio.set_frame_rate(target_sr).set_channels(1).set_sample_width(2)
    audio.export(out_path, format="wav")


def compute_embedding_for_wav(wav_path: str) -> np.ndarray:
    """
    Compute ECAPA embedding using ONNX model.
    """

    # ---------- LOAD AUDIO ----------
    audio, sr = librosa.load(wav_path, sr=16000, mono=True)

    # ---------- FBANK FEATURES ----------
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=16000,
        n_fft=400,
        hop_length=160,
        n_mels=80,
        fmin=20,
        fmax=7600,
        power=2.0,
    )

    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # shape → (frames, 80)
    features = log_mel.T.astype(np.float32)

    # ONNX expects batch dimension
    features = np.expand_dims(features, axis=0)

    # ---------- ONNX INFERENCE ----------
    embedding = onnx_session.run(
        None,
        {onnx_input_name: features}
    )[0]

    embedding = embedding.squeeze()

    # ---------- L2 NORMALIZE ----------
    embedding = embedding / (np.linalg.norm(embedding) + 1e-9)

    return embedding



def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


from fastapi import Form

@app.post("/enroll")
def enroll(
    org_id: str = Form(...),
    clips: List[UploadFile] = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):

    # ---------- AUTH ----------
    session_token = f"Bearer {credentials.credentials}"
    user = get_user_from_session(session_token)
    uid = user["id"]

    if len(clips) != 5:
        raise HTTPException(status_code=400, detail="Exactly 5 clips required")

    timestamp = int(time.time())
    embeddings = []

    # ---------- PROCESS CLIPS ----------
    for upload_file in clips:

        suffix = os.path.splitext(upload_file.filename)[1] or ".bin"

        file_id = str(uuid.uuid4())

        tmp_input_path = os.path.join(
            TEMP_AUDIO_DIR,
            f"{file_id}_input{suffix}"
        )

        tmp_wav_path = os.path.join(
            TEMP_AUDIO_DIR,
            f"{file_id}.wav"
        )

        try:
            # save uploaded audio
            with open(tmp_input_path, "wb") as f:
                f.write(upload_file.file.read())

            # convert → wav
            convert_to_wav(tmp_input_path, tmp_wav_path, 16000)

            # compute embedding (file guaranteed to exist)
            emb = compute_embedding_for_wav(tmp_wav_path)
            embeddings.append(emb)

        finally:
            # 🔒 cleanup AFTER embedding fully completes
            if os.path.exists(tmp_input_path):
                os.remove(tmp_input_path)

            if os.path.exists(tmp_wav_path):
                os.remove(tmp_wav_path)

    # ---------- AGGREGATE ----------
    emb_matrix = np.vstack(embeddings)
    mean_emb = emb_matrix.mean(axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-9)

    embedding_list = mean_emb.astype(float).tolist()

    # ---------- STORE ONLY EMBEDDING ----------
    record = {
        "uid": uid,
        "org_id": org_id,
        "embedding": embedding_list,
        "embedding_dim": len(embedding_list),
        "model": EMBEDDING_MODEL_NAME,
        "clip_count": len(embeddings),
        "classification_level": 1,
        "metadata": {
            "enrolled_at": timestamp,
            "model": EMBEDDING_MODEL_NAME,
        },
        "last_verified_at": None,
    }

    supabase.table("biometrics").upsert(
        record,
        on_conflict="uid,org_id"
    ).execute()

    return {
        "success": True,
        "model": EMBEDDING_MODEL_NAME,
        "embedding_dim": len(embedding_list),
    }