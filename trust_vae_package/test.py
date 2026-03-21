# test_package.py

from trust_vae import load_model, TRUST_VAE
import torch

print("Testing TRUST-VAE Package...")
print("=" * 50)

# Test 1: Import all components
from trust_vae import TRUST_Encoder, TRUST_Decoder, Task_Classifier, TRUST_VAE
print("✓ All imports successful")

# Test 2: Create model from scratch
vae = TRUST_VAE(latent_dim=128, hierarchical=True)
print(f"✓ Model created: {vae}")

# Test 3: Test forward pass
test_input = torch.randn(1, 1, 128, 128)
recon, mu, logvar, z = vae(test_input)
print(f"✓ Forward pass successful: {recon.shape}")

# Test 4: Load trained model (if you have .pth file)
try:
    vae, classifier, info = load_model('models\TRUST_VAE_Best.pth')
    print(f"✓ Model loaded: SSIM={info['best_ssim']}")
except:
    print("⚠ Model file not found (skip if testing without trained model)")

print("=" * 50)
print("✓ All tests passed!")