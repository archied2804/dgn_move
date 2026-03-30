"""

Download a dataset using the dgn.datasets.DatasetDownloader and inspect its contents. 
This dataset is used for training the Diffusion Graph Network (DGN) in the paper:

  "Learning Distributions of Complex Fluid Simulations with
   Diffusion Graph Networks"
  Mario Lino, Tobias Pfaff, Nils Thuerey
  ICLR 2025 (Oral)  |  https://arxiv.org/abs/2504.02843

Usage:
    python data_download.py --data_path /path/to/data

"""


# -----------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------

import os
import torch
from torchvision import transforms
import argparse

import dgn4cfd as dgn


# -----------------------------------------------------------------------
# Arg parsing and seeding for reproducibility
# -----------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Download Ellipse Dataset for DGN4CFD (Lino et al. ICLR 2025)."
)
parser.add_argument(
    "--gpu", type=int, default=-1,
    help="CUDA device index. Use -1 to force CPU."
)
parser.add_argument(
    "--nt", type=int, default=10,
    help="Number of time-steps in the training simulations."
)
args = parser.parse_args()

# Initial seed for reproducibility
seed = 0
torch.manual_seed(seed)

# file_system strategy avoids "too many open files" errors with multiple
# DataLoader workers on Linux / HPC systems. (This is CORRECT and necessary!)
torch.multiprocessing.set_sharing_strategy('file_system')

# -----------------------------------------------------------------------
# Device configuration
# -----------------------------------------------------------------------

# Choose GPU if available
device = (
    torch.device(f"cuda:{args.gpu}")
    if args.gpu >= 0 and torch.cuda.is_available()
    else torch.device("cpu")
)
print("  Ellipse DGN -- DGN4CFD (Lino et al. ICLR 2025), used for testing")
print(f"  Device          : {device}")
print(f"  PyTorch version : {torch.__version__}")

# -----------------------------------------------------------------------
# Dataset download
# -----------------------------------------------------------------------

_quick_ds = dgn.datasets.pOnEllipse(
    # The dataset can be downloaded from the web:
    path      = dgn.datasets.DatasetDownloader(dgn.datasets.DatasetUrl.pOnEllipseTrain).file_path,
    # Or you can directly provide the dataset path if already downloaded:
    # path    = "DATASET_PATH",
    T         = args.nt,
    transform = transforms.Compose([dgn.transforms.ScaleEdgeAttr(0.015)]),
)

_g = _quick_ds.get_sequence(0, sequence_start=0)
print(f"[Stage 1] Training sims  : {len(_quick_ds)}")
print(f"[Stage 1] Nodes per wing : {_g.num_nodes}")
print(f"[Stage 1] pos  shape     : {_g.pos.shape}    (x, y, z)")
print(f"[Stage 1] loc  shape     : {_g.loc.shape}    (nx, ny, nz)")
print(f"[Stage 1] target shape   : {_g.target.shape} (p at sampled t)")
del _quick_ds, _g

os.makedirs('./checkpoints', exist_ok=True)
os.makedirs('./boards', exist_ok=True)
