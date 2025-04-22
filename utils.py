import os
import torch
import config
from models import get_model

def load_model(name: str):
    """
    Instantiate and load a model’s saved weights from saved_models/{name}.pt
    """
    model = get_model(name, config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM)
    path  = os.path.join(config.MODELS_DIR, f"{name}.pt")
    model.load_state_dict(torch.load(path))
    return model
