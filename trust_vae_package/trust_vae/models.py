import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    """Xavier initialization for stability"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.01)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.01)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class TRUST_Encoder(nn.Module):
    def __init__(self, latent_dim=128, hierarchical=True):
        super(TRUST_Encoder, self).__init__()
        self.hierarchical = hierarchical
        self.latent_dim = latent_dim
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.flatten = nn.Flatten()
        
        if hierarchical:
            self.fc_mu_high = nn.Linear(512*8*8, latent_dim // 2)
            self.fc_logvar_high = nn.Linear(512*8*8, latent_dim // 2)
            self.fc_mu_low = nn.Linear(512*8*8, latent_dim // 2)
            self.fc_logvar_low = nn.Linear(512*8*8, latent_dim // 2)
        else:
            self.fc_mu = nn.Linear(512*8*8, latent_dim)
            self.fc_logvar = nn.Linear(512*8*8, latent_dim)
        
        self.apply(weights_init)
    
    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, -10, 0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        
        if self.hierarchical:
            mu_high = self.fc_mu_high(x)
            logvar_high = torch.clamp(self.fc_logvar_high(x), -10, 0)
            mu_low = self.fc_mu_low(x)
            logvar_low = torch.clamp(self.fc_logvar_low(x), -10, 0)
            z_high = self.reparameterize(mu_high, logvar_high)
            z_low = self.reparameterize(mu_low, logvar_low)
            z = torch.cat([z_high, z_low], dim=1)
            return (mu_high, mu_low), (logvar_high, logvar_low), z
        else:
            mu = self.fc_mu(x)
            logvar = torch.clamp(self.fc_logvar(x), -10, 0)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z


class TRUST_Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(TRUST_Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        self.apply(weights_init)
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 8, 8)
        x = self.deconv_layers(x)
        return x


class Task_Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(Task_Classifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(512, num_classes)
        self.apply(weights_init)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TRUST_VAE(nn.Module):
    def __init__(self, latent_dim=128, hierarchical=True):
        super(TRUST_VAE, self).__init__()
        self.encoder = TRUST_Encoder(latent_dim, hierarchical)
        self.decoder = TRUST_Decoder(latent_dim)
        self.latent_dim = latent_dim
        self.hierarchical = hierarchical
    
    def forward(self, x):
        mu, logvar, z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar, z
    
    def encode(self, x):
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode from latent space"""
        return self.decoder(z)
    
    def generate(self, z):
        """Generate image from latent vector"""
        return self.decode(z)