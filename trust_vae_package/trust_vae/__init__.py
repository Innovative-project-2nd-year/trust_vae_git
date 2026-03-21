"""
TRUST-VAE: Task-Aware, Uncertainty-Guided and Interpretable VAE
"""

from .models import TRUST_Encoder, TRUST_Decoder, Task_Classifier, TRUST_VAE
from .utils import load_model, load_from_pkl, save_model
from .version import __version__

__all__ = [
    'TRUST_Encoder',
    'TRUST_Decoder', 
    'Task_Classifier',
    'TRUST_VAE',
    'load_model',
    'load_from_pkl',
    'save_model',
    '__version__'
]

__version__ = '1.0.0'
__author__ = 'TRUST-VAE Team'