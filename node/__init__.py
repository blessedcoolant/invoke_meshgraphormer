from typing import OrderedDict

import numpy as np
import torch
from invokeai.app.services.config.config_default import get_config
from PIL import Image, ImageOps

from .detector import MeshGraphormer

config = get_config()


class MeshGraphormerDetector:
    def __init__(self, detector: MeshGraphormer, device=torch.device):
        self.detector = detector
        self.device = device

    @staticmethod
    def load_detector(
        mesh_graphormer_hand_model_dict: OrderedDict[str, torch.Tensor],
        hrnet_v2_model_path: OrderedDict[str, torch.Tensor],
    ) -> MeshGraphormer:
        detector = MeshGraphormer(mesh_graphormer_hand_model_dict, hrnet_v2_model_path)
        return detector

    def to(self):
        self.detector._model.to(self.device)
        self.detector.mano_model.to(self.device)
        self.detector.mano_model.layer.to(self.device)
        return self

    def __call__(self, image: Image.Image, mask_bbox_padding=30, **kwargs):
        image = np.array(image, dtype=np.uint8)
        image_height, image_width = image.shape[:2]

        depth_map, mask, _ = self.detector.get_depth(image, mask_bbox_padding)

        if depth_map is None:
            # Return blank images if no depth map is detected
            depth_map = np.zeros_like(image)
            mask = np.zeros_like(image)

        depth_map = Image.fromarray(depth_map)
        mask = ImageOps.invert(Image.fromarray(mask))

        return depth_map, mask
