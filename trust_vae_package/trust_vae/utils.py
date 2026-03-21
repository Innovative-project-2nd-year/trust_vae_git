import torch
import pickle
import os

def load_model(model_path, device='cpu', model_type='pth'):
    """
    Load TRUST-VAE model from .pth file
    
    Args:
        model_path: Path to .pth file
        device: Device to load model on ('cpu' or 'cuda')
        model_type: 'pth' for PyTorch checkpoint
    
    Returns:
        vae: TRUST_VAE model
        classifier: Task_Classifier model
        info: Model information dict
    """
    from .models import TRUST_VAE, Task_Classifier
    
    print(f"Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    vae = TRUST_VAE(latent_dim=128, hierarchical=True).to(device)
    classifier = Task_Classifier(num_classes=2).to(device)
    
    if 'vae_state_dict' in checkpoint:
        vae.load_state_dict(checkpoint['vae_state_dict'])
    else:
        vae.load_state_dict(checkpoint)
    
    if 'classifier_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    vae.eval()
    classifier.eval()
    
    info = {
        'best_ssim': checkpoint.get('best_ssim', 'N/A'),
        'epoch': checkpoint.get('epoch', 'N/A'),
        'latent_dim': 128,
        'classes': 2
    }
    
    print(f"✓ Model loaded successfully")
    print(f"  - Best SSIM: {info['best_ssim']}")
    print(f"  - Epoch: {info['epoch']}")
    
    return vae, classifier, info


def load_from_pkl(pkl_path, device='cpu'):
    """
    Load TRUST-VAE model from .pkl file
    
    Args:
        pkl_path: Path to .pkl file
        device: Device to load model on
    
    Returns:
        vae: TRUST_VAE model
        classifier: Task_Classifier model
        info: Model information dict
    """
    from .models import TRUST_VAE, Task_Classifier
    import numpy as np
    
    print(f"Loading model from {pkl_path}...")
    
    with open(pkl_path, 'rb') as f:
        model_data = pickle.load(f)
    
    vae = TRUST_VAE(latent_dim=model_data['latent_dim'], 
                    hierarchical=model_data['hierarchical']).to(device)
    classifier = Task_Classifier(num_classes=model_data['num_classes']).to(device)
    
    vae_state_dict = {
        k: torch.from_numpy(v).to(device) 
        for k, v in model_data['vae_state_dict'].items()
    }
    classifier_state_dict = {
        k: torch.from_numpy(v).to(device) 
        for k, v in model_data['classifier_state_dict'].items()
    }
    
    vae.load_state_dict(vae_state_dict)
    classifier.load_state_dict(classifier_state_dict)
    
    vae.eval()
    classifier.eval()
    
    info = {
        'best_ssim': model_data.get('best_ssim', 'N/A'),
        'epoch': model_data.get('epoch', 'N/A'),
        'latent_dim': model_data['latent_dim'],
        'classes': model_data['num_classes']
    }
    
    print(f"✓ Model loaded successfully")
    print(f"  - Best SSIM: {info['best_ssim']}")
    print(f"  - Epoch: {info['epoch']}")
    
    return vae, classifier, info


def save_model(vae, classifier, save_path, epoch=0, best_ssim=0, history=None):
    """
    Save TRUST-VAE model to .pth file
    
    Args:
        vae: TRUST_VAE model
        classifier: Task_Classifier model
        save_path: Path to save .pth file
        epoch: Current epoch number
        best_ssim: Best SSIM score
        history: Training history dict
    """
    torch.save({
        'epoch': epoch,
        'vae_state_dict': vae.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
        'best_ssim': best_ssim,
        'history': history
    }, save_path)
    
    print(f"✓ Model saved to {save_path}")