import hydra
from omegaconf import DictConfig
import time


@hydra.main(config_path="../../config", config_name="base_cfg", version_base=None)
def preprocess(cfg: DictConfig):
    print(f"Preprocessing data from: {cfg.dataset.root_dir}")
    time.sleep(5)
    # TODO: insert script that preprocess the raw dataset
    print(f"Preprocessing completed.")


if __name__ == "__main__":
    preprocess()
