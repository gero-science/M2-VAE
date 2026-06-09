import logging
import random
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from src.vae_models import VAE
from src.infra.seed import set_seed
from src.infra.logging import setup_logger
from src.infra.data import to_tensor

def calculate_kl_per_latent(mu, logvar):
    return -0.5 * np.mean(1 + logvar - mu**2 - np.exp(logvar), axis=0)


def extract_latents(model, loader, device):
    """
    Extract mu, logvar, sampled z for all patients
    """
    model.eval()

    mu_all, logvar_all, z_all = [], [], []

    local_gen = torch.Generator(device=device)
    local_gen.manual_seed(torch.seed() % (2**32))

    with torch.no_grad():
        for yb, xb in loader:
            yb = yb.to(device)
            xb = xb.to(device)

            # encoder returns mu, logvar
            mu, logvar = model.encoder(yb, xb)

            std = torch.exp(0.5 * logvar)
            eps = torch.randn(mu.shape, device=mu.device)
            z = mu + eps * std

            mu_all.append(mu.cpu())
            logvar_all.append(logvar.cpu())
            z_all.append(z.cpu())

    return (
        torch.cat(mu_all).numpy(),
        torch.cat(logvar_all).numpy(),
        torch.cat(z_all).numpy(),
    )


# Main
@hydra.main(config_path="configs", config_name="latents_config", version_base="1.3")
def main(cfg: DictConfig):
    # Output dir
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir / "latents.log")

    logger.info("Latent extraction started")
    logger.info("Config:\n" + OmegaConf.to_yaml(cfg))

    # Seed
    set_seed(cfg.seed)

    # Device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load data
    logger.info("Loading X/Y/uids...")

    X = np.load(Path(cfg.data.input_dir) / "X.npy")
    Y = np.load(Path(cfg.data.input_dir) / "Y.npy")
    uids = np.load(Path(cfg.data.input_dir) / "uids.npy", allow_pickle=True)

    X = to_tensor(X)
    Y = to_tensor(Y)

    dataset = TensorDataset(Y, X)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    D = Y.shape[1]
    C = X.shape[1]

    logger.info(f"Data loaded: N={len(dataset)}, D={D}, C={C}")

    # Loop over trained models
    kl_threshold = cfg.kl_threshold
    models_processed = 0

    for model_cfg in cfg.top_configs:
        name = model_cfg.name
        enc_layers = list(model_cfg.encoder_layers)
        dec_layers = enc_layers[::-1]
        latent_dim = model_cfg.latent_dim

        model_path = Path(model_cfg.model_path)

        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            continue

        logger.info(f"\nProcessing model: {name}")
        logger.info(f"Architecture: {enc_layers}")

        # Load model
        model = VAE(D, C, enc_layers, dec_layers, latent_dim).to(device)

        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        # Extract latents
        mu_all, logvar_all, z_all = extract_latents(model, loader, device)

        # KL per latent
        kl_vals = calculate_kl_per_latent(mu_all, logvar_all)

        informative_mask = kl_vals > kl_threshold
        informative_idx = np.where(informative_mask)[0]

        if len(informative_idx) == 0:
            logger.warning(f"{name}: no informative latents found")
            continue

        logger.info(
            f"{name}: {len(informative_idx)} informative latents (threshold={kl_threshold})"
        )

        # Keep only informative latents
        mu_inf = mu_all[:, informative_idx]
        z_inf = z_all[:, informative_idx]

        # Save parquet
        df = pd.DataFrame()

        for j in range(mu_inf.shape[1]):
            df[f"mu_latent_{j}"] = mu_inf[:, j]
            df[f"z_latent_{j}"] = z_inf[:, j]

        df["eid"] = uids
        df["model_name"] = name
        df["latent_indices"] = [informative_idx.tolist()] * len(df)
        df["kl_values"] = [kl_vals[informative_idx].tolist()] * len(df)

        out_path = out_dir / f"{name}_informative_latents.parquet"
        df.to_parquet(out_path, index=False)

        logger.info(f"Saved - {out_path}")
        models_processed += 1

    logger.info(f"\nDone. Models processed: {models_processed}")
    logger.info(f"Results saved in: {out_dir}")


if __name__ == "__main__":
    main()
