from src.data import bvFTDDataset
from torch.utils.data import WeightedRandomSampler


def get_dataset(cfg):
    if cfg.dataset.name == "bvFTD":
        train_dataset = bvFTDDataset(
            root_dir=cfg.dataset.root_dir,
            imgtype=cfg.dataset.imgtype,
            train=True,
            severity=cfg.dataset.severity,
            resize=cfg.dataset.resize,
        )
        val_dataset = bvFTDDataset(
            root_dir=cfg.dataset.root_dir,
            imgtype=cfg.dataset.imgtype,
            train=True,
            severity=cfg.dataset.severity,
            resize=cfg.dataset.resize,
        )
        sampler = None
        return train_dataset, val_dataset, sampler
    raise ValueError(f"{cfg.dataset.name} Dataset is not available")
