import os
import torch
import logging

from whisper.model import Whisper, ModelDimensions

MODEL_DIR = "./weights"
os.makedirs(MODEL_DIR, exist_ok=True)

RAW_WEIGHTS_DIR = "./raw_weights"

def process_weights():
    files = os.listdir(RAW_WEIGHTS_DIR)
    logging.info(f"All weights to process: {files}")

    for weights_file in os.listdir(RAW_WEIGHTS_DIR):

        with open(os.path.join(RAW_WEIGHTS_DIR, weights_file), "rb") as fp:
            logging.info(f"Processing {weights_file}")
            checkpoint = torch.load(fp, map_location="cpu")
            dims = ModelDimensions(**checkpoint["dims"])
            model = Whisper(dims)
            model.load_state_dict(checkpoint["model_state_dict"])

        torch.save(model, os.path.join(MODEL_DIR, weights_file))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    process_weights()