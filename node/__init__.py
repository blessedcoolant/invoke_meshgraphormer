import pathlib

import numpy as np
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.backend.util.devices import choose_torch_device
from invokeai.backend.util.util import download_with_progress_bar
from PIL import Image, ImageOps

from .detector import MeshGraphormer

config = InvokeAIAppConfig.get_config()

MESH_GRAPHORMER_MODEL_PATHS = {
    "graphormer_hand_state_dict.bin": {
        "url": "https://datarelease.blob.core.windows.net/metro/models/graphormer_hand_state_dict.bin?download=true",
        "local": "any/annotators/mesh_graphormer/graphormer_hand_state_dict.bin",
    },
    "hrnetv2_w64_imagenet_pretrained.pth": {
        "url": "https://datarelease.blob.core.windows.net/metro/models/hrnetv2_w64_imagenet_pretrained.pth",
        "local": "any/annotators/mesh_graphormer/hrnetv2_w64_imagenet_pretrained.pth",
    },
}


class MeshGraphormerDetector:
    def __init__(self, detector):
        self.detector = detector

    @classmethod
    def load_detector(cls):
        MESH_GRAPHORMER_HAND_MODEL = pathlib.Path(
            config.models_path / MESH_GRAPHORMER_MODEL_PATHS["graphormer_hand_state_dict.bin"]["local"]
        )
        if not MESH_GRAPHORMER_HAND_MODEL.exists():
            download_with_progress_bar(
                MESH_GRAPHORMER_MODEL_PATHS["graphormer_hand_state_dict.bin"]["url"], MESH_GRAPHORMER_HAND_MODEL
            )

        HRNET_V2_MODEL = pathlib.Path(
            config.models_path / MESH_GRAPHORMER_MODEL_PATHS["hrnetv2_w64_imagenet_pretrained.pth"]["local"]
        )
        if not HRNET_V2_MODEL.exists():
            download_with_progress_bar(
                MESH_GRAPHORMER_MODEL_PATHS["hrnetv2_w64_imagenet_pretrained.pth"]["url"], HRNET_V2_MODEL
            )

        hand_checkpoint = MESH_GRAPHORMER_HAND_MODEL.as_posix()
        hrnet_checkpoint = HRNET_V2_MODEL.as_posix()
        detector = MeshGraphormer(hand_checkpoint, hrnet_checkpoint)
        return cls(detector)

    def to(self):
        device = choose_torch_device()
        self.detector._model.to(device)
        self.detector.mano_model.to(device)
        self.detector.mano_model.layer.to(device)
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
