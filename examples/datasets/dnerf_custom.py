import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from datasets.dnerf_synthetic import SubjectLoader

from .utils import Rays


def _load_custom_renderings(root_fp: str, split: str):
    """Load images from disk."""
    if not root_fp.startswith("/"):
        # allow relative path. e.g., "./data/dnerf_synthetic/"
        root_fp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            root_fp,
        )

    data_dir = os.path.join(root_fp)
    
    images = []
    camtoworlds = []
    timestamps = []
    focal = 0
    
    if split == "train":
        # Load camera matrix
        camera_matrix_path = os.path.join(data_dir, "main/K.txt")
        K = np.loadtxt(camera_matrix_path)
        fx = K[0,0]
        fy = K[1,1]
        assert fx == fy
        focal = fx

        # Load camera poses
        poses_path = os.path.join(data_dir, "main/poses.txt")
        with open(poses_path) as f:
            poses = np.fromstring(f.read(), dtype=np.float32, sep=" ")
            poses = poses.reshape((-1, 4, 4))
            poses = torch.from_numpy(poses)

        # Make sure it fits in memory
        start = 0
        end = 450
        step = 1
        # poses = poses[start:end:step]

        # Load the images
        for img_id in range(start, end, step):
            image_path = os.path.join(data_dir, f"main/camera_{str(img_id).zfill(4)}.jpeg")
            rgba = imageio.imread(image_path)
            # Add 0 to alpha channel
            h, w, _ = rgba.shape
            rgba = np.concatenate((rgba, np.ones((h, w, 1)) * 255), axis=2)
            
            images.append(rgba)
            camtoworlds.append(poses[img_id])
            timestamps.append((1/30) * img_id)
    elif split == "test":
        # Load camera matrix
        camera_matrix_path = os.path.join(data_dir, "perspective_1/K.txt")
        K = np.loadtxt(camera_matrix_path)
        fx = K[0,0]
        fy = K[1,1]
        assert fx == fy
        focal = fx
        
        for p in range(1, 5):
            # Load camera poses
            poses_path = os.path.join(data_dir, "main/poses.txt")
            with open(poses_path) as f:
                poses = np.fromstring(f.read(), dtype=np.float32, sep=" ")
                poses = poses.reshape((-1, 4, 4))
                poses = torch.from_numpy(poses)
                
            # Load image
            image_path = os.path.join(data_dir, f"perspective_{p}/camera_0000.jpeg")
            rgba = imageio.imread(image_path)
            # Add 0 to alpha channel
            h, w, _ = rgba.shape
            rgba = np.concatenate((rgba, np.zeros((h, w, 1)) * 255), axis=2)
            
            images.append(rgba)
            camtoworlds.append(poses[0])
            timestamps.append(0)
    
    
    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)
    timestamps = np.stack(timestamps, axis=0)

    return images, camtoworlds, focal, timestamps


class CustomLoader(SubjectLoader):
    """Custom subject data loader for training and evaluation."""

    SPLITS = ["train", "val", "test"]

    WIDTH, HEIGHT = 720, 1280
    NEAR, FAR = 0.1, 200.0
    OPENGL_CAMERA = False

    def __init__(
        self,
        root_fp: str,
        split: str,
        color_bkgd_aug: str = "white",
        num_rays: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
        device: str = "cpu",
    ):
        # super().__init__()
        assert split in self.SPLITS, "%s" % split
        assert color_bkgd_aug in ["white", "black", "random"]
        self.split = split
        self.num_rays = num_rays
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far
        self.training = (num_rays is not None) and (
            split in ["train", "trainval"]
        )
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        (
            self.images,
            self.camtoworlds,
            self.focal,
            self.timestamps,
        ) = _load_custom_renderings(root_fp, split)
        self.images = torch.from_numpy(self.images).to(device).to(torch.uint8)
        self.camtoworlds = (
            torch.from_numpy(self.camtoworlds).to(device).to(torch.float32)
        )
        self.timestamps = (
            torch.from_numpy(self.timestamps)
            .to(device)
            .to(torch.float32)[:, None]
        )
        self.K = torch.tensor(
            [
                [self.focal, 0, self.WIDTH / 2.0],
                [0, self.focal, self.HEIGHT / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
            device=device,
        )  # (3, 3)
        assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)
