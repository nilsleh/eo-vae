import importlib
from omegaconf import OmegaConf

def enum_resolver(mod: str, enum_name: str, member: str):
    module = importlib.import_module(mod)
    return getattr(getattr(module, enum_name), member)