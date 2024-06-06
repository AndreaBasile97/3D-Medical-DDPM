import hydra
from omegaconf import DictConfig
import time
import os


@hydra.main(config_path="../../config", config_name="base_cfg", version_base=None)
def preprocess(cfg: DictConfig):
    print(f"Preprocessing data from: {cfg.dataset.raw_dir_bvFTD}")
    time.sleep(5)
    print(f"Preprocessing completed. Data saved on {cfg.dataset.bvFTD_processed_dir}")
    time.sleep(5)
    print(f"Preprocessing data from: {cfg.dataset.raw_dir_HC}")
    print(f"Preprocessing completed. Data saved on {cfg.dataset.HC_processed_dir}")


if __name__ == "__main__":
    preprocess()
