import argparse
import pathlib
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from datasets.nerf_synthetic import SubjectLoader
from lpips import LPIPS
from radiance_fields.mlp import VanillaNeRFRadianceField

import os
import sys
base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from examples.utils import (
    NERF_SYNTHETIC_SCENES,
    render_image_with_occgrid,
    set_random_seed,
)
from nerfacc.estimators.occ_grid import OccGridEstimator

device = "cuda:0"
set_random_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    default=str(pathlib.Path.cwd() / "data/nerf_synthetic"),
    help="the root dir of the dataset",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=["train", "trainval"],
    help="which train split to use",
)
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="the path of the pretrained model",
)
parser.add_argument(
    "--scene",
    type=str,
    default="lego",
    choices=NERF_SYNTHETIC_SCENES,
    help="which scene to use",
)
parser.add_argument(
    "--test_chunk_size",
    type=int,
    default=4096,
)
args = parser.parse_args()

test_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split="test",
    num_rays=None,
    device=device,
)

# scene parameters
aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
near_plane = 0.0
far_plane = 1.0e10
# model parameters
grid_resolution = 128
grid_nlvl = 1
# render parameters
render_step_size = 5e-3


radiance_field = VanillaNeRFRadianceField().to(device)
estimator = OccGridEstimator(
    roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
).to(device)

if args.model_path is not None:
    checkpoint = torch.load(args.model_path)
    radiance_field.load_state_dict(checkpoint["radiance_field_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    estimator.load_state_dict(checkpoint["estimator_state_dict"])
    step = checkpoint["step"]
else:
    step = 0


lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

radiance_field.eval()
estimator.eval()

psnrs = []
lpips = []
with torch.no_grad():
    for i in tqdm.tqdm(range(len(test_dataset))):
        data = test_dataset[i]
        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]
        # rendering
        rgb, acc, depth, _ = render_image_with_occgrid(
            radiance_field,
            estimator,
            rays,
            # rendering options
            near_plane=near_plane,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            # test options
            test_chunk_size=args.test_chunk_size,
        )

        mse = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(mse) / np.log(10.0)
        psnrs.append(psnr.item())
        lpips.append(lpips_fn(rgb, pixels).item())
        if i == 0 or True:
            imageio.imwrite(
                f"novel_views/rgb_test_{i}.png",
                (rgb.cpu().numpy() * 255).astype(np.uint8),
            )
            imageio.imwrite(
                f"novel_views/rgb_error_{i}.png",
                (
                    (rgb - pixels).norm(dim=-1).cpu().numpy() * 255
                ).astype(np.uint8),
            )
psnr_avg = sum(psnrs) / len(psnrs)
lpips_avg = sum(lpips) / len(lpips)
print(f"evaluation: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}")