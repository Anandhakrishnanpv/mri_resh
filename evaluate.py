import torch, time
from utils import compute_psnr
from skimage.metrics import structural_similarity as ssim

def evaluate(model, loader, device):
    psnr_in = psnr_out = ssim_in = ssim_out = 0
    count = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            for i in range(x.shape[0]):
                xi, yi, pi = x[i,0], y[i,0], pred[i,0]
                psnr_in += compute_psnr(xi, yi).item()
                psnr_out += compute_psnr(pi, yi).item()
                ssim_in += ssim(xi.cpu(), yi.cpu(), data_range=1.0)
                ssim_out += ssim(pi.cpu(), yi.cpu(), data_range=1.0)
                count += 1

    return psnr_in/count, psnr_out/count, ssim_in/count, ssim_out/count

def inference_time(model, x, runs=50):
    for _ in range(10): model(x)
    start = time.time()
    for _ in range(runs): model(x)
    return (time.time()-start)/runs*1000
