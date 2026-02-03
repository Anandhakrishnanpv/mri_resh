import h5py, numpy as np, torch
from torch.utils.data import Dataset
from utils import center_crop, random_mask

class FastMRISliceDataset(Dataset):
    def __init__(self, files, crop_size=256, accel=4):
        self.slices = []
        self.crop_size = crop_size
        self.accel = accel

        for f in files:
            with h5py.File(f, "r") as hf:
                for i in range(hf["kspace"].shape[0]):
                    self.slices.append((f, i))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        f, i = self.slices[idx]
        with h5py.File(f, "r") as hf:
            k = hf["kspace"][i]

        if k.ndim == 3:
            k = k[...,0] + 1j*k[...,1]

        mask = random_mask(k.shape, self.accel)
        img_und = np.abs(np.fft.ifft2(k * mask))
        img_gt  = np.abs(np.fft.ifft2(k))

        img_und = center_crop(img_und, 256, 256)
        img_gt  = center_crop(img_gt, 256, 256)

        img_und /= img_und.max() + 1e-8
        img_gt  /= img_gt.max() + 1e-8

        return (
            torch.tensor(img_und).float().unsqueeze(0),
            torch.tensor(img_gt).float().unsqueeze(0)
        )
