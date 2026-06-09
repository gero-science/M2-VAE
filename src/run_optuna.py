import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.vae_models import VAE, vae_loss
from src.infra.seed import set_seed
from src.infra.logging import setup_logger
from src.infra.data import to_tensor


def worker_init_fn(worker_id):
    # Ensures reproducibility across DataLoader workers
    # Each worker gets unique but deterministic seed: base_seed + worker_id
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Optuna objective
def build_objective(train_ds, test_ds, D, C, device, logger):
    def objective(trial):
        torch.cuda.empty_cache()

        logger.info(f"Trial {trial.number} started")

        latent_dim = trial.suggest_int("latent_dim", 20, 40)
        n_layers = trial.suggest_int("n_layers", 2, 2)

        # Wide range of layer sizes from small to large
        # Small sizes for baseline, large sizes for better GPU utilization
        possible_sizes = [64, 128, 256, 512, 1024, 2048]
        encoder_layers = [
            trial.suggest_categorical(f"enc_layer_{i}", possible_sizes) for i in range(n_layers)
        ]
        decoder_layers = encoder_layers[::-1]

        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        # Full range of batch sizes from reasonable minimum to maximum for 23GB GPU
        # Small batches for smaller models, large batches to maximize GPU usage
        batch_size = trial.suggest_categorical("batch_size", [8192, 8192])
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

        logger.info(
            f"Trial {trial.number} params: "
            f"latent={latent_dim}, layers={encoder_layers}, "
            f"lr={lr:.2e}, bs={batch_size}, wd={weight_decay:.2e}"
        )

        # Aggressive DataLoader optimization for maximum GPU utilization
        # High num_workers + prefetch_factor keeps GPU fed with data
        # pin_memory + non_blocking transfers enable async CPU→GPU pipeline
        num_workers = 6 if device.type == "cuda" else 2
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if device.type == "cuda" else False,
            worker_init_fn=worker_init_fn,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=4 if num_workers > 0 else None,  # Prefetch 4 batches per worker
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if device.type == "cuda" else False,
            worker_init_fn=worker_init_fn,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=4 if num_workers > 0 else None,
        )

        try:
            model = VAE(D, C, encoder_layers, decoder_layers, latent_dim).to(device)

            yb, xb = next(iter(train_loader))
            yb, xb = yb[:8].to(device), xb[:8].to(device)
            model(yb, xb)
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning(f"Trial {trial.number} pruned due to OOM")
                torch.cuda.empty_cache()
                raise optuna.TrialPruned()
            raise

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Mixed precision training for better GPU utilization
        # Uses tensor cores on modern GPUs, ~2x speedup with minimal accuracy impact
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        max_epochs = 200
        patience = 10
        best_val_loss = float("inf")
        no_improve = 0

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()

            # Train
            model.train()
            for yb, xb in train_loader:
                # non_blocking allows async CPU→GPU transfer with pinned memory
                yb, xb = yb.to(device, non_blocking=True), xb.to(device, non_blocking=True)
                optimizer.zero_grad()

                # Mixed precision forward pass
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        logits, mu, logvar = model(yb, xb)
                        loss, _, _ = vae_loss(yb, logits, mu, logvar)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits, mu, logvar = model(yb, xb)
                    loss, _, _ = vae_loss(yb, logits, mu, logvar)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            # Validation
            # loss.item() is average per sample, so multiply by batch size to get total loss for batch
            # Then divide by total samples (not batches) to get correct overall average
            # Impact is small if last batch is only slightly smaller, but still technically correct
            model.eval()
            total_val_loss = 0.0
            total_samples = 0
            with torch.no_grad():
                for yb, xb in test_loader:
                    # non_blocking allows async CPU→GPU transfer with pinned memory
                    yb, xb = yb.to(device, non_blocking=True), xb.to(device, non_blocking=True)
                    logits, mu, logvar = model(yb, xb)
                    loss, _, _ = vae_loss(yb, logits, mu, logvar)
                    batch_size = yb.size(0)
                    total_val_loss += loss.item() * batch_size
                    total_samples += batch_size
            total_val_loss /= total_samples

            dt = time.time() - t0

            # Log every 20 epochs
            if epoch == 1 or epoch % 20 == 0:
                logger.info(
                    f"Trial {trial.number} | Epoch {epoch:03d} | "
                    f"Val loss {total_val_loss:.4f} | "
                    f"{dt:.1f}s"
                )

            # Early stopping
            trial.report(total_val_loss, epoch)
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned by Optuna")
                raise optuna.TrialPruned()

            if total_val_loss < best_val_loss - 1e-4:
                best_val_loss = total_val_loss
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                logger.info(f"Trial {trial.number} early stopped at epoch {epoch}")
                break

        logger.info(f"Trial {trial.number} finished | Best val {best_val_loss:.4f}")

        # Clean up model to free GPU memory before next trial
        # del removes Python references, allowing GC to free GPU memory
        # empty_cache() then makes freed memory available for next trial
        # Without explicit del, objects may linger until GC runs, causing OOM after many trials
        del model
        del optimizer
        torch.cuda.empty_cache()

        return best_val_loss

    return objective


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Experiment folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{cfg.experiment_name}_{timestamp}"
    exp_dir = Path(cfg.output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Logger
    logger = setup_logger(exp_dir / "train.log")

    logger.info("Experiment started")
    logger.info("Config:\n" + OmegaConf.to_yaml(cfg))

    # Seed
    set_seed(cfg.seed)

    # Load data
    logger.info("Loading data...")

    X = np.load(Path(cfg.input_dir) / "X.npy")
    Y = np.load(Path(cfg.input_dir) / "Y.npy")

    X = to_tensor(X)
    Y = to_tensor(Y)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Data shape: X={X.shape}, Y={Y.shape}")

    # Dataset & split
    dataset = TensorDataset(Y, X)
    N = len(dataset)
    generator = torch.Generator().manual_seed(cfg.seed)
    test_size = int(cfg.test_split * N)
    trainval_size = N - test_size

    trainval_ds, holdout_test_ds = random_split(
        dataset, [trainval_size, test_size], generator=generator
    )

    val_size = int(cfg.val_split * trainval_size)
    train_size = trainval_size - val_size

    optuna_train_ds, optuna_val_ds = random_split(
        trainval_ds, [train_size, val_size], generator=generator
    )

    logger.info(
        f"Splits: "
        f"train={len(optuna_train_ds)}, "
        f"val={len(optuna_val_ds)}, "
        f"test={len(holdout_test_ds)}"
    )

    D = Y.shape[1]
    C = X.shape[1]

    # Optuna
    logger.info("Starting Optuna optimization")

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.seed + 1)
    )

    objective = build_objective(optuna_train_ds, optuna_val_ds, D, C, device, logger)
    study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=False)

    # Save study and best params
    import joblib

    joblib.dump(study, exp_dir / "study.pkl")

    best_params = study.best_trial.params
    with open(exp_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    logger.info("Optuna finished")
    logger.info(f"Best params: {best_params}")

    # Final training
    logger.info("Starting final training")

    latent_dim = best_params["latent_dim"]
    n_layers = best_params["n_layers"]
    lr = best_params["lr"]
    batch_size = best_params["batch_size"]
    weight_decay = best_params["weight_decay"]

    encoder_layers = [best_params[f"enc_layer_{i}"] for i in range(n_layers)]
    decoder_layers = encoder_layers[::-1]

    # Aggressive DataLoader optimization for maximum GPU utilization
    # High num_workers + prefetch_factor keeps GPU fed with data
    # pin_memory + non_blocking transfers enable async CPU→GPU pipeline
    num_workers = 6 if device.type == "cuda" else 2
    final_train_ds = torch.utils.data.ConcatDataset([optuna_train_ds, optuna_val_ds])
    train_loader = DataLoader(
        final_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
        worker_init_fn=worker_init_fn,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,  # Prefetch 4 batches per worker
    )
    test_loader = DataLoader(
        holdout_test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
        worker_init_fn=worker_init_fn,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    model = VAE(D, C, encoder_layers, decoder_layers, latent_dim).to(device)

    # Compile model for faster execution (PyTorch 2.0+)
    # JIT compilation optimizes computation graph, reduces Python overhead
    # Disabled: requires triton which may not be installed
    # if device.type == "cuda" and hasattr(torch, 'compile'):
    #     try:
    #         model = torch.compile(model, mode='reduce-overhead')
    #     except Exception:
    #         pass  # Fall back to eager mode if compilation fails

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Mixed precision training for better GPU utilization
    # Uses tensor cores on modern GPUs, ~2x speedup with minimal accuracy impact
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history = []

    for epoch in range(1, cfg.final_epochs):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        train_bce = 0.0
        train_kl = 0.0
        total_train_samples = 0
        for yb, xb in train_loader:
            # non_blocking allows async CPU→GPU transfer with pinned memory
            yb, xb = yb.to(device, non_blocking=True), xb.to(device, non_blocking=True)
            optimizer.zero_grad()

            # Mixed precision forward pass
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits, mu, logvar = model(yb, xb)
                    loss, bce, kl = vae_loss(yb, logits, mu, logvar)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, mu, logvar = model(yb, xb)
                loss, bce, kl = vae_loss(yb, logits, mu, logvar)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = yb.size(0)
            train_loss += loss.item() * batch_size
            train_bce += bce.item() * batch_size
            train_kl += kl.item() * batch_size
            total_train_samples += batch_size
        train_loss /= total_train_samples
        train_bce /= total_train_samples
        train_kl /= total_train_samples

        # Validation
        # Fix: need to weight by batch size since last batch might be smaller
        # Otherwise averaging by number of batches gives wrong result when batch sizes vary
        model.eval()
        val_loss, val_bce, val_kl = 0.0, 0.0, 0.0
        total_val_samples = 0
        with torch.no_grad():
            for yb, xb in test_loader:
                # non_blocking allows async CPU→GPU transfer with pinned memory
                yb, xb = yb.to(device, non_blocking=True), xb.to(device, non_blocking=True)
                logits, mu, logvar = model(yb, xb)
                loss, bce, kl = vae_loss(yb, logits, mu, logvar)
                batch_size = yb.size(0)
                val_loss += loss.item() * batch_size
                val_bce += bce.item() * batch_size
                val_kl += kl.item() * batch_size
                total_val_samples += batch_size

        val_loss /= total_val_samples
        val_bce /= total_val_samples
        val_kl /= total_val_samples

        # Track more metrics: BCE and KL separately to understand model behavior
        # BCE measures reconstruction quality, KL measures how close latent space is to prior
        history.append([epoch, train_loss, train_bce, train_kl, val_loss, val_bce, val_kl])

        dt = time.time() - t0

        # Log every 50 epochs
        if epoch == 1 or epoch % 50 == 0:
            logger.info(
                f"Epoch {epoch:04d}/{cfg.final_epochs} | "
                f"Train {train_loss:.4f} (BCE {train_bce:.4f}, KL {train_kl:.4f}) | "
                f"Val {val_loss:.4f} (BCE {val_bce:.4f}, KL {val_kl:.4f}) | "
                f"{dt:.1f}s"
            )

        # Save checkpoint periodically to avoid losing progress if training crashes
        # Saves every 100 epochs or at the end
        if epoch % 100 == 0 or epoch == cfg.final_epochs:
            checkpoint_path = exp_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                },
                checkpoint_path,
            )
            logger.info(f"Checkpoint saved at epoch {epoch}")

    # Save results
    torch.save(model.state_dict(), exp_dir / "final_model.pt")

    hist_df = pd.DataFrame(
        history,
        columns=["epoch", "train_loss", "train_bce", "train_kl", "val_loss", "val_bce", "val_kl"],
    )
    hist_df.to_csv(exp_dir / "history_of_training.csv", index=False)

    # Plot loss curves with separate BCE and KL components for better insight
    # Helps understand if model is overfitting (BCE) or regularization issues (KL)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(hist_df["epoch"], hist_df["train_loss"], label="Train Loss")
    axes[0].plot(hist_df["epoch"], hist_df["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].legend()
    axes[0].set_title("Total Loss")

    axes[1].plot(hist_df["epoch"], hist_df["train_bce"], label="Train BCE", linestyle="--")
    axes[1].plot(hist_df["epoch"], hist_df["val_bce"], label="Val BCE", linestyle="--")
    axes[1].plot(hist_df["epoch"], hist_df["train_kl"], label="Train KL", linestyle=":")
    axes[1].plot(hist_df["epoch"], hist_df["val_kl"], label="Val KL", linestyle=":")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss Component")
    axes[1].legend()
    axes[1].set_title("BCE and KL Components")

    plt.tight_layout()
    plt.savefig(exp_dir / "loss_curve.png")
    plt.close()

    config = OmegaConf.to_container(cfg, resolve=True)
    config["best_params"] = best_params

    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    logger.info("Experiment finished successfully")
    logger.info(f" Results saved in: {exp_dir}")


if __name__ == "__main__":
    main()
