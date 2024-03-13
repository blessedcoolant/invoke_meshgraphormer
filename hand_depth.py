import gc

import torch
from invokeai.app.invocations.fields import FieldDescriptions
from invokeai.app.invocations.primitives import ImageField
from invokeai.backend.util.devices import choose_torch_device
from invokeai.invocation_api import (
    BaseInvocation,
    BaseInvocationOutput,
    InputField,
    InvocationContext,
    OutputField,
    WithMetadata,
    invocation,
    invocation_output,
)
from PIL import Image

from .node import MeshGraphormerDetector

meshgraphormer_detector = None


@invocation_output("meshgraphormer_output")
class HandDepthOutput(BaseInvocationOutput):
    """Base class for to output Meshgraphormer results"""

    image: ImageField = OutputField(description="Improved hands depth map")
    mask: ImageField = OutputField(description="Hands area mask")
    width: int = OutputField(description="The width of the depth map in pixels")
    height: int = OutputField(description="The height of the depth map in pixels")


@invocation(
    "hand_depth_mesh_graphormer_image_processor",
    title="Hand Depth w/ MeshGraphormer",
    tags=["controlnet", "depth", "mesh graphormer", "hand refiner", "preprocessor"],
    category="controlnet",
    version="1.0.1",
)
class HandDepthMeshGraphormerProcessor(BaseInvocation, WithMetadata):
    """Generate hand depth maps to inpaint with using ControlNet"""

    image: ImageField = InputField(description="The image to process")
    resolution: int = InputField(default=512, ge=64, multiple_of=64, description=FieldDescriptions.image_res)
    mask_padding: int = InputField(default=30, ge=0, description="Amount to pad the hand mask by")
    offload: bool = InputField(default=False, description="Offload model after usage")

    def run_processor(self, image: Image.Image):
        global meshgraphormer_detector

        if not meshgraphormer_detector:
            meshgraphormer_detector = MeshGraphormerDetector.load_detector()

        if image.mode == "RGBA":
            image = image.convert("RGB")

        image_width, image_height = image.size

        # Resize before sending for processing
        new_height = int(image_height * (self.resolution / image_width))
        image = image.resize((self.resolution, new_height))

        processed_image, mask = meshgraphormer_detector(image=image, mask_bbox_padding=self.mask_padding)

        if self.offload:
            meshgraphormer_detector = None
            gc.collect()
            if choose_torch_device() == "cuda":
                torch.cuda.empty_cache()

        return processed_image, mask

    def invoke(self, context: InvocationContext) -> HandDepthOutput:
        raw_image = context.images.get_pil_image(self.image.image_name)
        processed_image, mask = self.run_processor(raw_image)

        image_dto = context.images.save(processed_image)
        mask_dto = context.images.save(mask)

        processed_image_field = ImageField(image_name=image_dto.image_name)
        processed_mask_field = ImageField(image_name=mask_dto.image_name)

        return HandDepthOutput(
            image=processed_image_field,
            mask=processed_mask_field,
            width=image_dto.width,
            height=image_dto.height,
        )
