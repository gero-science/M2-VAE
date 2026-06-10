import os
import random
import json
import logging
import uuid
import numpy as np
import torch
import hydra
from pathlib import Path

from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.vae_model_bmi import VAE_BMI, vae_bmi_loss
from src.infra.seed import set_seed
from src.infra.logging import setup_logger
from src.infra.data import to_tensor


def load_np(path):
    return np.load(path, mmap_mode="r")


def make_run_id():
    return f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


# data
def load_data(X_path, Y_path, BMI_path):

    X = load_np(X_path)
    Y = load_np(Y_path)
    bmi = load_np(BMI_path)
    bmi = bmi.reshape(-1, 1).astype(np.float32)

    Y = (Y > 0.5).astype(np.float32)

    X = np.hstack([X, bmi.reshape(-1, 1)])

    return X, Y, bmi


def train(cfg):

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id = make_run_id()

    output_dir = os.path.join(cfg.output_dir, run_id)

    model_dir = os.path.join(output_dir, "model")
    log_dir = os.path.join(output_dir, "logs")
    metrics_dir = os.path.join(output_dir, "metrics")
    hydra_dir = os.path.join(output_dir, "hydra")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(hydra_dir, exist_ok=True)

    logger = setup_logger(os.path.join(log_dir, "train.log"))

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # run metadata
    with open(os.path.join(output_dir, "run.json"), "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "seed": cfg.seed,
            },
            f,
            indent=2,
        )

    # data
    X, Y, bmi = load_data(cfg.data.X_path, cfg.data.Y_path, cfg.data.BMI_path)

    dataset = TensorDataset(to_tensor(Y), to_tensor(X), to_tensor(bmi))

    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    D = Y.shape[1]
    C = X.shape[1]

    model = VAE_BMI(
        D, C, cfg.model.encoder_layers, cfg.model.decoder_layers, cfg.model.latent_dim
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )

    history = []
    best_loss = float("inf")

    # Training loop
    for epoch in range(1, cfg.training.max_epochs + 1):
        model.train()
        total_loss = 0.0

        for yb, xb, bmb in loader:
            yb = yb.to(device)
            xb = xb.to(device)
            bmb = bmb.to(device)

            optimizer.zero_grad()

            logits_y, bmi_pred, mu, logvar, log_sigma2 = model(yb, xb)

            loss, _, _, _ = vae_bmi_loss(
                yb, logits_y, bmb, bmi_pred, mu, logvar, log_sigma2
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        history.append(avg_loss)

        logger.info(f"[{run_id}] Epoch {epoch} | loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(model_dir, "best.pt"))

    metrics = {
        "best_loss": float(best_loss),
        "final_loss": float(history[-1]),
        "epochs": cfg.training.max_epochs,
        "run_id": run_id,
    }

    with open(os.path.join(metrics_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(os.path.join(metrics_dir, "history.npy"), np.array(history))

    logger.info(f"Training finished | run_id={run_id}")


CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


@hydra.main(
    version_base=None,
    config_path=CONFIG_DIR,
    config_name="base-config",
)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
