import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import itertools
from lpips import LPIPS

import os
import sys
base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from examples.utils import render_image_with_propnet, set_random_seed
from radiance_fields.ngp import NGPDensityField, NGPRadianceField
from nerfacc.estimators.prop_net import PropNetEstimator
from datasets.unbounded_custom import SubjectLoader


device = "cuda:0"
set_random_seed(42)
max_steps = 5000
init_batch_size = 4096
weight_decay = 0.0
# scene parameters
unbounded = True
aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
near_plane = 0.2  # TODO: Try 0.02
far_plane = 1e3

proposal_networks = [
    NGPDensityField(
        aabb=aabb,
        unbounded=unbounded,
        n_levels=5,
        max_resolution=128,
    ).to(device),
    NGPDensityField(
        aabb=aabb,
        unbounded=unbounded,
        n_levels=5,
        max_resolution=256,
    ).to(device),
]

prop_optimizer = torch.optim.Adam(
    itertools.chain(
        *[p.parameters() for p in proposal_networks],
    ),
    lr=1e-2,
    eps=1e-15,
    weight_decay=weight_decay,
)
prop_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    [
        torch.optim.lr_scheduler.LinearLR(
            prop_optimizer, start_factor=0.01, total_iters=100
        ),
        torch.optim.lr_scheduler.MultiStepLR(
            prop_optimizer,
            milestones=[
                max_steps // 2,
                max_steps * 3 // 4,
                max_steps * 9 // 10,
            ],
            gamma=0.33,
        ),
    ]
)

radiance_field = NGPRadianceField(aabb=aabb, unbounded=unbounded).to(device)
estimator = PropNetEstimator(prop_optimizer, prop_scheduler).to(device)

checkpoint = torch.load("ngp_nerf_prop.pt")
radiance_field.load_state_dict(checkpoint["radiance_field_state_dict"])
prop_optimizer.load_state_dict(checkpoint["prop_optimizer_state_dict"])
prop_scheduler.load_state_dict(checkpoint["prop_scheduler_state_dict"])
estimator.load_state_dict(checkpoint["estimator_state_dict"])
proposal_networks[0].load_state_dict(checkpoint["proposal_network_0_state_dict"])
proposal_networks[1].load_state_dict(checkpoint["proposal_network_1_state_dict"])

num_samples = 48
num_samples_per_prop = [256, 96]
sampling_type = "lindisp"
opaque_bkgd = True
test_dataset_kwargs = {"factor": 4}

lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()


test_dataset = SubjectLoader(
    subject_id="garden",
    root_fp="../nerf/dataset/baseline/",
    split="test",
    num_rays=None,
    device=device,
    **test_dataset_kwargs
)

# evaluation
radiance_field.eval()
for p in proposal_networks:
    p.eval()
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
        rgb, acc, depth, _, = render_image_with_propnet(
            radiance_field,
            proposal_networks,
            estimator,
            rays,
            # rendering options
            num_samples=num_samples,
            num_samples_per_prop=num_samples_per_prop,
            near_plane=near_plane,
            far_plane=far_plane,
            sampling_type=sampling_type,
            opaque_bkgd=opaque_bkgd,
            render_bkgd=render_bkgd,
            # test options
            test_chunk_size=8192,
        )
        mse = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(mse) / np.log(10.0)
        psnrs.append(psnr.item())
        lpips.append(lpips_fn(rgb, pixels).item())
        if i == 0 or True:
            imageio.imwrite(
                f"novel_views_unbounded/rgb_test_{i}.png",
                (rgb.cpu().numpy() * 255).astype(np.uint8),
            )
            imageio.imwrite(
                f"novel_views_unbounded/rgb_error_{i}.png",
                (
                    (rgb - pixels).norm(dim=-1).cpu().numpy() * 255
                ).astype(np.uint8),
            )
            # break
psnr_avg = sum(psnrs) / len(psnrs)
lpips_avg = sum(lpips) / len(lpips)
print(f"evaluation: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}")