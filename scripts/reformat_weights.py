import os
import torch

from whisper.model import Whisper, ModelDimensions

MODEL_DIR = "./weights"
os.makedirs(MODEL_DIR, exist_ok=True)

for weights_file in os.listdir("./raw_weights"):

    with open(weights_file, "rb") as fp:
        checkpoint = torch.load(fp, map_location="cpu")
        dims = ModelDimensions(**checkpoint["dims"])
        model = Whisper(dims)
        model.load_state_dict(checkpoint["model_state_dict"])

    torch.save(model, os.path.join(MODEL_DIR, weights_file))
