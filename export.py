import torch
from speechbrain.pretrained import EncoderClassifier

print("Loading ECAPA...")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/ecapa"
)

embedding_model = classifier.mods.embedding_model
embedding_model.eval()

# ECAPA expects FBANK features:
# shape = (batch, frames, 80)

dummy_features = torch.randn(1, 200, 80)

print("Exporting ECAPA embedding network...")

torch.onnx.export(
    embedding_model,
    dummy_features,
    "ecapa_embedding.onnx",
    input_names=["features"],
    output_names=["embedding"],
    dynamic_axes={
        "features": {1: "frames"},
        "embedding": {0: "batch"},
    },
    opset_version=17,
)

print("✅ Export successful")