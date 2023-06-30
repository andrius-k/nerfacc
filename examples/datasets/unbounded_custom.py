"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import collections
import os
import sys

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from .utils import Rays

_PATH = os.path.abspath(__file__)

sys.path.insert(
    0, os.path.join(os.path.dirname(_PATH), "..", "pycolmap", "pycolmap")
)
from scene_manager import SceneManager


def _load_camera_poses(root_fp: str, fp: str):
    poses_path = os.path.join(root_fp, fp)

    # Load actor poses path
    actor_poses_path =  poses_path.replace("poses.txt", "actor.txt")
    with open(actor_poses_path) as f:
        actor_poses = np.fromstring(f.read(), dtype=np.float32, sep=" ")
        actor_poses = actor_poses.reshape((-1, 4, 4))

    with open(poses_path) as f:
        poses = np.fromstring(f.read(), dtype=np.float32, sep=" ")
        poses = poses.reshape((-1, 4, 4))

        # poses[:,0,3] -= 1000
        # poses[:,1,3] += 155
        # poses[:,2,3] += 350

        # Center the scene
        poses[:,0,3] -= actor_poses[:,0,3]
        poses[:,1,3] -= actor_poses[:,1,3]
        poses[:,2,3] -= actor_poses[:,2,3]

        # Normalize the scene to be between -1 and 1
        scale = np.max(np.abs(poses[:,:3,3]))
        poses[:,0,3] /= scale
        poses[:,1,3] /= scale
        poses[:,2,3] /= scale

        poses = torch.from_numpy(poses)

    return poses


def _load_dataset(root_fp: str, training: bool, factor: int = 1):
    images = []
    camtoworlds = []
    timestamps = []
    K = []

    if training:
        # for perspective in ["main"]:
        for perspective in ["main", "perspective_1", "perspective_2"]:
            # Load camera matrix
            camera_matrix_path = os.path.join(root_fp, f"{perspective}/K.txt")
            K = np.loadtxt(camera_matrix_path)

            # Load camera poses
            poses = _load_camera_poses(root_fp, f"{perspective}/poses.txt")

            # Make sure it fits in memory
            start = 0
            end = 450
            step = 1
            # Load the images
            for img_id in range(start, end, step):
                image_path = os.path.join(root_fp, f"{perspective}/camera_{str(img_id).zfill(4)}.jpeg")
                rgb = imageio.imread(image_path)
                images.append(rgb)
                camtoworlds.append(poses[img_id])
                # timestamps.append((1/30) * img_id)
                timestamps.append(float(img_id) / (len(poses) - 1))
    else:
        # Load camera matrix
        camera_matrix_path = os.path.join(root_fp, "perspective_1/K.txt")
        K = np.loadtxt(camera_matrix_path)

        # Load some images from the training dataset
        poses = _load_camera_poses(root_fp, "main/poses.txt")
        for img_id in range(10):
            image_path = os.path.join(root_fp, f"main/camera_{str(img_id).zfill(4)}.jpeg")
            rgb = imageio.imread(image_path)
            images.append(rgb)
            camtoworlds.append(poses[img_id])
            # timestamps.append((1/30) * img_id)
            timestamps.append(float(img_id) / (len(poses) - 1))
        
        for p in range(2, 5):
            # Load camera poses
            poses = _load_camera_poses(root_fp, f"perspective_{p}/poses.txt")

            # Load image
            image_path = os.path.join(root_fp, f"perspective_{p}/camera_0000.jpeg")
            rgb = imageio.imread(image_path)
            images.append(rgb)
            camtoworlds.append(poses[0])
            timestamps.append(0)

    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)
    timestamps = np.stack(timestamps, axis=0)

    # Convert extrinsics to camera-to-world
    # camtoworlds = np.linalg.inv(camtoworlds)

    return images, camtoworlds, timestamps, K


def _load_baseline_dataset(root_fp: str, training: bool, factor: int = 1):
    images = []
    camtoworlds = []
    timestamps = []
    K = []

    # Load camera matrix
    camera_matrix_path = os.path.join(root_fp, "K.txt")
    K = np.loadtxt(camera_matrix_path)

    # Load camera poses
    poses = _load_camera_poses(root_fp, "poses.txt")

    if training:
        start = 0
        end = 350
        step = 1
    else:
        start = 350
        end = 400
        step = 1
    
    # Load the images
    for img_id in range(start, end, step):
        image_path = os.path.join(root_fp, f"camera_{str(img_id).zfill(4)}.jpeg")
        # image_path = os.path.join(root_fp, f"{str(img_id).zfill(4)}.jpeg")
        rgb = imageio.imread(image_path)
        images.append(rgb)
        camtoworlds.append(poses[img_id])
        timestamps.append(0)

    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)
    timestamps = np.stack(timestamps, axis=0)

    # Convert extrinsics to camera-to-world
    # camtoworlds = np.linalg.inv(camtoworlds)

    return images, camtoworlds, timestamps, K


def similarity_from_cameras(c2w, strict_scaling):
    """
    reference: nerf-factory
    Get a similarity transform to normalize dataset
    from c2w (OpenCV convention) cameras
    :param c2w: (N, 4)
    :return T (4,4) , scale (float)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    # we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        # In the unlikely case the original data has y+ up axis,
        # rotate 180-deg about x axis
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    #  R_align = np.eye(3) # DEBUG
    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene using camera center rays
    # find the closest point to the origin for each camera's center ray
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds

    # median for more robustness
    translate = -np.median(nearest, axis=0)

    #  translate = -np.mean(t, axis=0)  # DEBUG

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))

    return transform, scale


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "test"]
    SUBJECT_IDS = [
        "garden",
        "bicycle",
        "bonsai",
        "counter",
        "kitchen",
        "room",
        "stump",
        "lego",
    ]

    OPENGL_CAMERA = False

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        color_bkgd_aug: str = "white",
        num_rays: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
        factor: int = 1,
        device: str = "cpu",
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        assert color_bkgd_aug in ["white", "black", "random"]
        self.split = split
        self.num_rays = num_rays
        self.near = near
        self.far = far
        self.training = (num_rays is not None) and (
            split in ["train", "trainval"]
        )
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images

        # self.images, self.camtoworlds, self.timestamps, self.K = _load_baseline_dataset(
        #     root_fp, self.training, factor
        # )

        self.images, self.camtoworlds, self.timestamps, self.K = _load_dataset(
            root_fp, self.training, factor
        )

        # normalize the scene
        # T, sscale = similarity_from_cameras(
        #     self.camtoworlds, strict_scaling=False
        # )
        # self.camtoworlds = np.einsum("nij, ki -> nkj", self.camtoworlds, T)
        # self.camtoworlds[:, :3, 3] *= sscale
        # to tensor
        self.images = torch.from_numpy(self.images).to(torch.uint8).to(device)
        self.camtoworlds = (
            torch.from_numpy(self.camtoworlds).to(torch.float32).to(device)
        )
        self.timestamps = (
            torch.from_numpy(self.timestamps)
            .to(device)
            .to(torch.float32)[:, None]
        )

        self.K = torch.tensor(self.K).to(torch.float32).to(device)
        self.height, self.width = self.images.shape[1:3]


    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        pixels, rays = data["rgb"], data["rays"]

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, device=self.images.device)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.images.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.images.device)
        else:
            # just use white during inference
            color_bkgd = torch.ones(3, device=self.images.device)

        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgb", "rays"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays

        if self.training:
            if self.batch_over_images:
                image_id = torch.randint(
                    0,
                    len(self.images),
                    size=(num_rays,),
                    device=self.images.device,
                )
            else:
                image_id = [index] * num_rays
            x = torch.randint(
                0, self.width, size=(num_rays,), device=self.images.device
            )
            y = torch.randint(
                0, self.height, size=(num_rays,), device=self.images.device
            )
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.width, device=self.images.device),
                torch.arange(self.height, device=self.images.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        rgb = self.images[image_id, y, x] / 255.0  # (num_rays, 3)
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        timestamps = self.timestamps[image_id]

        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5)
                    / self.K[1, 1]
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [num_rays, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            rgb = torch.reshape(rgb, (num_rays, 3))
        else:
            origins = torch.reshape(origins, (self.height, self.width, 3))
            viewdirs = torch.reshape(viewdirs, (self.height, self.width, 3))
            rgb = torch.reshape(rgb, (self.height, self.width, 3))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgb": rgb,  # [h, w, 3] or [num_rays, 3]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            "timestamps": timestamps,  # [num_rays, 1]
        }
