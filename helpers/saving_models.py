import torch

def save_model(model, path):
    """Save the weights of model to path"""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Load weights from path model"""
    try:
        model.load_state_dict(torch.load(path))
    except:
        print("Failed to load model weights from {}.".format(path))
