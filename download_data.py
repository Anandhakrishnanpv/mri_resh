from huggingface_hub import snapshot_download
import os

ROOT = os.path.expanduser("~/Anandhakrishnan P V")
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

snapshot_download(
    repo_id="FLowOak/mri_knee",
    repo_type="dataset",
    local_dir=DATA,
    local_dir_use_symlinks=False
)

print("✅ Dataset downloaded")
