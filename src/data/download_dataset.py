import hydra
from omegaconf import DictConfig
import time
import os


@hydra.main(config_path="../../config", config_name="base_cfg", version_base=None)
def download_dataset(cfg: DictConfig):
    print(f"Downloading data from: {cfg.dataset.download_url_bvFTD}")
    time.sleep(5)
    # TODO: insert script that downloads the dataset
    print(f"Download completed.")
    print(f"Downloading data from: {cfg.dataset.download_url_HC}")
    time.sleep(5)
    # TODO: insert script that downloads the dataset
    print(f"Download completed.")


if __name__ == "__main__":
    download_dataset()
