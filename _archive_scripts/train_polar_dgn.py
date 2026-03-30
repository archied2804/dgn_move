"""
Train a PolarDiffusionGraphNet to generate ellipse surface geometries
conditioned on pressure fields and Re.

Run:
    python train_polar_dgn.py --experiment_id 0 --gpu 0
"""

import torch
from torchvision import transforms
import argparse
import sys
sys.path.insert(0, '../..')

import dgn4cfd as dgn
from dgn4cfd.nn.diffusion.polar_diffusion_process import PolarRadialDiffusionProcess
from dgn4cfd.nn.diffusion.models.polar_dgn import PolarDiffusionGraphNet
from dgn4cfd.nn.losses import PolarGeomHybridLoss
from dgn4cfd.datasets import pOnEllipseGeometry

torch.multiprocessing.set_sharing_strategy('file_system')

argparser = argparse.ArgumentParser()
argparser.add_argument('--experiment_id', type=int, default=0)
argparser.add_argument('--gpu',           type=int, default=0)
args = argparser.parse_args()

seed = 0
torch.manual_seed(seed)

experiment = {
    0: {
        'name':   'polar_geom_dgn',
        'epochs': 100,
        'depths': [2,2,2,2],
        'width':  128,
        'nt':     10,   # how many pressure timesteps to average over,
        'diffusion_steps': 1000,
    },
    1: {
        'name':   'polar_geom_dgn_small',
        'epochs': 500,
        'depths': [4,2,1,2,4],
        'width':  64,
        'nt':     10,
        'diffusion_steps': 200,
    },
}[args.experiment_id]

# ----------------------------------------------------------------- settings
train_settings = dgn.nn.TrainingSettings(
    name          = experiment['name'],
    folder        = './checkpoints',
    tensor_board  = './boards',
    chk_interval  = 1,
    #training_loss = PolarGeomHybridLoss(vlb_weight=0.001),
    training_loss = dgn.nn.losses.HybridLoss(),
    epochs        = 100,
    batch_size    = 32,
    lr            = 1e-5,
    grad_clip     = {"epoch": 0, "limit": 1},
    scheduler     = {"factor": 0.1, "patience": 50, "loss": 'training'},
    stopping      = 1e-8,
    step_sampler  = dgn.nn.diffusion.ImportanceStepSampler,
    device        = torch.device(f'cuda:{args.gpu}') if args.gpu >= 0
                    else torch.device('cpu'),
)

# ---------------------------------------------------------------- transform
transform = transforms.Compose([
    dgn.transforms.MeshEllipse(),           # build graph from clean positions
    dgn.transforms.ScaleEdgeAttr(0.02),     # normalise Δpos edge features
    dgn.transforms.MeshCoarsening(
        num_scales      =  len(experiment['depths']),

        #rel_pos_scaling = [0.02, 0.06,],
        #rel_pos_scaling = [0.02, 0.06, 0.15],
        #rel_pos_scaling = [0.02, 0.06, 0.15, 0.3],
        rel_pos_scaling = [0.02, 0.06, 0.15, 0.3, 0.6],   # scale factor for relative position edge features at each scale

        scalar_rel_pos  = True,
    ),
])

# ----------------------------------------------------------------- dataset
dataset = pOnEllipseGeometry(
    path      = "/home/m22729ad/Projects-Local/DGN_Move/pOnEllipseTrain.h5",
    T         = experiment['nt'],
    transform = transform,
    preload   = False,
)
dataloader = dgn.DataLoader(
    dataset     = dataset,
    batch_size  = train_settings['batch_size'],
    shuffle     = True,
    num_workers = 8,
)

# -------------------------------------------------------- diffusion process
diffusion_process = PolarRadialDiffusionProcess(
    num_steps     = experiment['diffusion_steps'],
    schedule_type = 'cosine',   # linear, cosine, ...?
)

# ------------------------------------------------------------- architecture
# in_node_features  : 2  (noisy x, y  — the diffused positions)
# cond_node_features: 4  (p, nx, ny from graph.loc  +  Re from graph.glob)
# cond_edge_features: 2  (Δx, Δy between noisy neighbours)

arch = {
    'in_node_features':   2,   # noisy (x, y)
    'cond_node_features': 4,   # [p_norm, nx, ny]  +  [Re]
    'cond_edge_features': 2,   # relative position  (updated per step in loss)
    'depths':             experiment['depths'],
    'fnns_width':         experiment['width'],
    'aggr':               'sum',
    'dropout':            0.1,
}

model = PolarDiffusionGraphNet(
    diffusion_process  = diffusion_process,
    learnable_variance = True,
    arch               = arch,
)

# ----------------------------------------------------------------- training
model.fit(train_settings, dataloader)
