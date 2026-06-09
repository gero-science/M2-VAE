import torch
import torch.nn as nn


# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, cov_dim, hidden_dims, latent_dim):
        super().__init__()
        layers = []
        prev = input_dim + cov_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.mu = nn.Linear(prev if layers else (input_dim + cov_dim), latent_dim)
        self.logvar = nn.Linear(prev if layers else (input_dim + cov_dim), latent_dim)

    def forward(self, y, x):
        h = torch.cat([y, x], dim=1)
        h = self.net(h)
        return self.mu(h), self.logvar(h)


# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.out = nn.Linear(prev if layers else latent_dim, output_dim)

    def forward(self, z):
        h = self.net(z)
        logits = self.out(h)
        return logits


# VAE
class VAE(nn.Module):
    def __init__(self, input_dim, cov_dim, enc_layers, dec_layers, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, cov_dim, enc_layers, latent_dim)
        self.decoder = Decoder(latent_dim, dec_layers, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, y, x):
        mu, logvar = self.encoder(y, x)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)
        return logits, mu, logvar


# Loss function
bce_logits = nn.BCEWithLogitsLoss(reduction="sum")


def vae_loss(y_true, logits, mu, logvar):
    bce = bce_logits(logits, y_true)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = (bce + kl) / y_true.size(0)
    return loss, bce / y_true.size(0), kl / y_true.size(0)
