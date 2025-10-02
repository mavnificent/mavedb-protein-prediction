import torch.nn as nn
from typing import Type, Dict

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_model(cls: Type[nn.Module]) -> Type[nn.Module]:
    """Decorator to register a model by class name."""
    MODEL_REGISTRY[cls.__name__] = cls
    return cls