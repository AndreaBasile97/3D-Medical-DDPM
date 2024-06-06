import hydra
from omegaconf import DictConfig
import time


@hydra.main(config_path="../../config", config_name="base_cfg", version_base=None)
def preprocess(cfg: DictConfig):
    print(f"Preprocessing data from: {cfg.bvFTD_dataset.root_dir}")
    time.sleep(5)
    print(
        f"Preprocessing completed. Data saved on {cfg.processed_dataset.root_dir_bvFTD_processed}"
    )
    time.sleep(5)
    print(f"Preprocessing data from: {cfg.HC_dataset.root_dir}")
    print(
        f"Preprocessing completed. Data saved on {cfg.processed_dataset.root_dir_hc_processed}"
    )


if __name__ == "__main__":
    preprocess()
