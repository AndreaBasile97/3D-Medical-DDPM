from re import I
import argparse
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
import os
import time
import random
import mlflow
import mlflow.pytorch
import dagshub

# To launch the script with a specified command:
# NCCL_P2P_DISABLE=1 accelerate launch train/train_ddpm.py


@hydra.main(config_path="../../config", config_name="base_cfg", version_base=None)
def run(cfg: DictConfig):
    dagshub.init("3D-Medical-DDPM", "andreabasile97", mlflow=True)
    # Check if CUDA is available and set the GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.model.gpus)
        print(f"Using GPU: {cfg.model.gpus}")
    else:
        print("CUDA is not available. Using CPU.")

    print(f"Using dataset from: {cfg.dataset.download_url_bvFTD}")
    print(f"Batch size: {cfg.model.batch_size}")
    # print(
    #     f"Model configuration: {cfg.model.hidden_layers} hidden layers, "
    #     f"{cfg.model.hidden_units} units per layer, learning rate: {cfg.model.learning_rate}"
    # )

    # Start an MLflow run
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_param("batch_size", cfg.model.batch_size)
        mlflow.log_param("gpu", cfg.model.gpus)
        # Log any other parameters from the config as needed
        # Example:
        # mlflow.log_param("hidden_layers", cfg.model.hidden_layers)
        # mlflow.log_param("hidden_units", cfg.model.hidden_units)
        # mlflow.log_param("learning_rate", cfg.model.learning_rate)

        epochs = 5  # Number of training epochs

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            time.sleep(0.5)  # Simulating training time

            # Simulate loss and accuracy values
            loss = random.uniform(0.1, 1.0)
            accuracy = random.uniform(50.0, 100.0)

            print(f"Loss: {loss:.4f} - Accuracy: {accuracy:.2f}%")

            # Log metrics
            mlflow.log_metric("loss", loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)

        print("Training completed.")

        # Save model (if any) using MLflow's PyTorch support
        # Example:
        # model = ...  # Your trained model
        # mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    run()

    # wandb.finish()

    # Incorporate GAN loss in DDPM training?
    # Incorporate GAN loss in UNET segmentation?
    # Maybe better if I don't use ema updates?
    # Use with other vqgan latent space (the one with more channels?)
