import torch
from PIAUNet.model.model import PhysicsInformedAttentionUNet

def load_model(path, device):
    model = PhysicsInformedAttentionUNet(in_ch=3, out_ch=2)
    state = torch.load(path, map_location=device)
    
    # Handle new checkpoint format with nested keys
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    
    model.to(device)
    model.eval()
    return model