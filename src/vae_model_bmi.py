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
        self.mu = nn.Linear(prev, latent_dim)
        self.logvar = nn.Linear(prev, latent_dim)

    def forward(self, y, x):
        h = torch.cat([y, x], dim=1)
        h = self.net(h)
        return self.mu(h), self.logvar(h)


# Decoder (predicts Y and BMI)
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h

        self.net = nn.Sequential(*layers) if layers else nn.Identity()

        self.out_y = nn.Linear(prev, output_dim)

        # New decoder head (predict BMI)
        self.out_bmi = nn.Linear(prev, 1)

    def forward(self, z):
        h = self.net(z)
        logits_y = self.out_y(h)
        bmi_pred = self.out_bmi(h)
        return logits_y, bmi_pred


# Full VAE with BMI
class VAE_BMI(nn.Module):
    def __init__(self, input_dim, cov_dim, enc_layers, dec_layers, latent_dim):
        super().__init__()

        self.encoder = Encoder(input_dim, cov_dim, enc_layers, latent_dim)
        self.decoder = Decoder(latent_dim, dec_layers, input_dim)

        # Learnable global log(sigma^2) for BMI
        self.log_sigma2_bmi = nn.Parameter(torch.tensor(0.0))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, y, x):
        mu, logvar = self.encoder(y, x)
        z = self.reparameterize(mu, logvar)
        logits_y, bmi_pred = self.decoder(z)
        return logits_y, bmi_pred, mu, logvar, self.log_sigma2_bmi


# Loss
bce_logits = nn.BCEWithLogitsLoss(reduction="sum")


def vae_bmi_loss(y_true, logits_y, bmi_true, bmi_pred, mu, logvar, log_sigma2_bmi):

    # Standard VAE parts
    bce = bce_logits(logits_y, y_true)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # BMI Gaussian:
    sigma2 = torch.exp(log_sigma2_bmi)
    inv_sigma2 = torch.exp(-log_sigma2_bmi)

    bmi_nll = 0.5 * torch.sum((bmi_true - bmi_pred) ** 2 * inv_sigma2 + log_sigma2_bmi)

    # Normalize by batch size
    batch = y_true.size(0)
    loss = (bce + kl + bmi_nll) / batch

    return loss, bce / batch, kl / batch, bmi_nll / batch
