from re import I
import argparse
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
import os
import time
import random

# NCCL_P2P_DISABLE=1 accelerate launch train/train_ddpm.py


@hydra.main(config_path="../config", config_name="base_cfg", version_base=None)
def run(cfg: DictConfig):
    torch.cuda.set_device(cfg.model.gpus)
    print(f"Using dataset from: {cfg.dataset.path}")
    print(f"Batch size: {cfg.dataset.batch_size}")
    print(
        f"Model configuration: {cfg.model.hidden_layers} hidden layers, "
        f"{cfg.model.hidden_units} units per layer, learning rate: {cfg.model.learning_rate}"
    )

    epochs = 5  # Numero di epoche di addestramento

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        time.sleep(0.5)  # Simulazione del tempo di addestramento

        # Simulazione dei valori di perdita e accuratezza
        loss = random.uniform(0.1, 1.0)
        accuracy = random.uniform(50.0, 100.0)

        print(f"Loss: {loss:.4f} - Accuracy: {accuracy:.2f}%")

    print("Training completed.")


if __name__ == "__main__":
    run()

    # wandb.finish()

    # Incorporate GAN loss in DDPM training?
    # Incorporate GAN loss in UNET segmentation?
    # Maybe better if I don't use ema updates?
    # Use with other vqgan latent space (the one with more channels?)
