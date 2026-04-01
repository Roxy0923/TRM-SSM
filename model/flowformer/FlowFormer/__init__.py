import torch
def build_flowformer(cfg, input_dim=3):
    name = cfg.transformer 
    if name == 'latentcostformer':
        from .LatentCostFormer.transformer import FlowFormer
    else:
        raise ValueError(f"FlowFormer = {name} is not a valid architecture!")

    return FlowFormer(cfg[name], input_dim=input_dim)