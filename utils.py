import os
import torch
from models import get_model
import config

def save_model(model, name):
    torch.save(model.state_dict(), os.path.join(config.MODELS_DIR, f"{name}.pt"))

def load_model(name):
    model = get_model(name, config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM)
    model.load_state_dict(torch.load(os.path.join(config.MODELS_DIR, f"{name}.pt")))
    model.eval()
    return model
