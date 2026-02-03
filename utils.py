import numpy as np, torch, random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def center_crop(img, h, w):
    H, W = img.shape
    sh, sw = (H-h)//2, (W-w)//2
    return img[sh:sh+h, sw:sw+w]

def random_mask(shape, accel):
    H, W = shape
    mask = np.zeros((H,W), np.float32)
    center = int(W*0.08)
    c = W//2
    mask[:, c-center//2:c+center//2] = 1

    prob = (W/accel - center) / W
    for i in range(W):
        if np.random.rand() < prob:
            mask[:,i] = 1
    return mask

def compute_psnr(pred, target):
    mse = torch.mean((pred-target)**2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
