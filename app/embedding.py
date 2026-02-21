from fastapi import UploadFile
from speechbrain.pretrained import SpeakerRecognition
from tempfile import NamedTemporaryFile
import numpy as np
import os
import subprocess
import torch

model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/ecapa"
)


async def extract_embedding(upload: UploadFile):
    # save uploaded file to tmp
    with NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
        tmp.write(await upload.read())
        tmp.flush()

        # convert anything → wav 16kHz mono
        wav = tmp.name + ".wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp.name, "-ac", "1", "-ar", "16000", wav],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        waveform = model.load_audio(wav)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # embed
        embedding = model.encode_batch(waveform)
        embedding = embedding.squeeze().detach().cpu().numpy().tolist()

        # cleanup
        os.remove(tmp.name)
        os.remove(wav)

        return embedding
