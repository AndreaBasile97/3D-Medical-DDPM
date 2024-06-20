from torch.utils.data.dataset import Dataset


class bvFTDDataset(Dataset):
    def __init__(self, root_dir):
        print(root_dir)
