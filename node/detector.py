import pathlib
import random
from argparse import Namespace
from pathlib import Path
from typing import OrderedDict

import cv2
import mediapipe as mp
import numpy as np
import torch
from invokeai.app.services.config.config_default import get_config
from invokeai.backend.util.devices import choose_torch_device
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from torchvision import transforms
from transformers.models.bert.configuration_bert import BertConfig
from trimesh import Trimesh
from trimesh.ray.ray_triangle import RayMeshIntersector

from ..mesh_graphormer.modeling._mano import MANO, Mesh
from ..mesh_graphormer.modeling.bert import Graphormer_Hand_Network as Graphormer_Network
from ..mesh_graphormer.modeling.bert.modeling_graphormer import Graphormer
from ..mesh_graphormer.modeling.hrnet.config import config as hrnet_config
from ..mesh_graphormer.modeling.hrnet.config import update_config as hrnet_update_config
from ..mesh_graphormer.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat

args = Namespace(
    num_hidden_layers=4,
    hidden_size=-1,
    num_attention_heads=4,
    intermediate_size=-1,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MeshGraphormer:
    def __init__(
        self,
        hand_checkpoint: OrderedDict[str, torch.Tensor],
        hrnet_checkpoint: OrderedDict[str, torch.Tensor],
        args=args,
    ) -> None:
        self.invoke_config = get_config()
        set_seed(88)
        self.device = choose_torch_device()

        # Initiate MANO and Mesh
        mano_model = MANO().to(self.device)
        mano_model.layer = mano_model.layer.to(self.device)
        mesh_sampler = Mesh(device=self.device)

        # Load pretrained model
        trans_encoder = []

        input_feat_dim = [int(item) for item in [2051, 512, 128]]
        hidden_feat_dim = [int(item) for item in [1024, 256, 64]]
        output_feat_dim = input_feat_dim[1:] + [3]

        # which encoder block to have graph convs
        which_blk_graph = [int(item) for item in [0, 0, 1]]

        for i in range(len(output_feat_dim)):
            config_class, model_class = BertConfig, Graphormer
            config = config_class.from_pretrained(
                pathlib.Path(
                    self.invoke_config.root_path.as_posix()
                    / self.invoke_config.custom_nodes_dir
                    / "invoke_meshgraphormer/mesh_graphormer/modeling/bert/bert-base-uncased/"
                ).as_posix()
            )

            config.output_attentions = False
            config.img_feature_dim = input_feat_dim[i]
            config.output_feature_dim = output_feat_dim[i]
            args.hidden_size = hidden_feat_dim[i]
            args.intermediate_size = int(args.hidden_size * 2)

            if which_blk_graph[i] == 1:
                config.graph_conv = True
            else:
                config.graph_conv = False

            config.mesh_type = "hand"

            # update model structure if specified in arguments
            update_params = ["num_hidden_layers", "hidden_size", "num_attention_heads", "intermediate_size"]
            for idx, param in enumerate(update_params):
                arg_param = getattr(args, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    setattr(config, param, arg_param)

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config)
            trans_encoder.append(model)

        hrnet_yaml = Path(__file__).parent / "data/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
        hrnet_update_config(hrnet_config, hrnet_yaml)
        backbone = get_cls_net_gridfeat(hrnet_config, pretrained_dict=hrnet_checkpoint)

        trans_encoder = torch.nn.Sequential(*trans_encoder)

        # build end-to-end Graphormer network (CNN backbone + multi-layer Graphormer encoder)
        _model = Graphormer_Network(args, config, backbone, trans_encoder)
        _model.load_state_dict(hand_checkpoint, strict=False)

        # update configs to enable attention outputs
        setattr(_model.trans_encoder[-1].config, "output_attentions", True)
        setattr(_model.trans_encoder[-1].config, "output_hidden_states", True)
        _model.trans_encoder[-1].bert.encoder.output_attentions = True
        _model.trans_encoder[-1].bert.encoder.output_hidden_states = True
        for iter_layer in range(4):
            _model.trans_encoder[-1].bert.encoder.layer[iter_layer].attention.self.output_attentions = True
        for inter_block in range(3):
            setattr(_model.trans_encoder[-1].config, "device", self.device)

        _model.to(self.device)
        self._model = _model
        self.mano_model = mano_model
        self.mesh_sampler = mesh_sampler

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        # Fix File loading is not yet supported on Windows
        with open(str(Path(__file__).parent / "data/hand_landmarker.task"), "rb") as file:
            model_data = file.read()
        base_options = python.BaseOptions(model_asset_buffer=model_data)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            num_hands=2,
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

    def get_rays(self, W, H, fx, fy, cx, cy, c2w_t, center_pixels):  # rot = I

        j, i = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32))
        if center_pixels:
            i = i.copy() + 0.5
            j = j.copy() + 0.5

        directions = np.stack([(i - cx) / fx, (j - cy) / fy, np.ones_like(i)], -1)
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

        rays_o = np.expand_dims(c2w_t, 0).repeat(H * W, 0)

        rays_d = directions  # (H, W, 3)
        rays_d = (rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)).reshape(-1, 3)

        return rays_o, rays_d

    def get_mask_bounding_box(self, extrema, H, W, padding=30, dynamic_resize=0.15):
        x_min, x_max, y_min, y_max = extrema
        bb_xpad = max(int((x_max - x_min + 1) * dynamic_resize), padding)
        bb_ypad = max(int((y_max - y_min + 1) * dynamic_resize), padding)
        bbx_min = np.max((x_min - bb_xpad, 0))
        bbx_max = np.min((x_max + bb_xpad, W - 1))
        bby_min = np.max((y_min - bb_ypad, 0))
        bby_max = np.min((y_max + bb_ypad, H - 1))
        return bbx_min, bbx_max, bby_min, bby_max

    def run_inference(self, img, Graphormer_model, mano, mesh_sampler, scale, crop_len):
        global args
        H, W = int(crop_len), int(crop_len)
        Graphormer_model.eval()
        mano.eval()
        device = next(Graphormer_model.parameters()).device
        with torch.no_grad():
            img_tensor = self.transform(img)
            batch_imgs = torch.unsqueeze(img_tensor, 0).to(device)

            # forward-pass
            pred_camera, pred_3d_joints, pred_vertices_sub, pred_vertices, hidden_states, att = Graphormer_model(
                batch_imgs, mano, mesh_sampler
            )

            # obtain 3d joints, which are regressed from the full mesh
            pred_3d_joints_from_mesh = mano.get_3d_joints(pred_vertices)

            pred_camera = pred_camera.cpu()
            pred_vertices = pred_vertices.cpu()
            mesh = Trimesh(vertices=pred_vertices[0], faces=mano.face)
            res = crop_len
            focal_length = 1000 * scale
            camera_t = np.array([-pred_camera[1], -pred_camera[2], -2 * focal_length / (res * pred_camera[0] + 1e-9)])
            pred_3d_joints_camera = pred_3d_joints_from_mesh.cpu()[0] - camera_t
            z_3d_dist = pred_3d_joints_camera[:, 2].clone()

            pred_2d_joints_img_space = (
                (pred_3d_joints_camera / z_3d_dist[:, None]) * np.array((focal_length, focal_length, 1))
            )[:, :2] + np.array((W / 2, H / 2))

            rays_o, rays_d = self.get_rays(W, H, focal_length, focal_length, W / 2, H / 2, camera_t, True)
            coords = np.array(list(np.ndindex(H, W))).reshape(H, W, -1).transpose(1, 0, 2).reshape(-1, 2)
            intersector = RayMeshIntersector(mesh)
            points, index_ray, _ = intersector.intersects_location(rays_o, rays_d, multiple_hits=False)

            tri_index = intersector.intersects_first(rays_o, rays_d)

            tri_index = tri_index[index_ray]

            assert len(index_ray) == len(tri_index)

            discriminator = np.sum(mesh.face_normals[tri_index] * rays_d[index_ray], axis=-1) <= 0
            points = points[discriminator]  # ray intesects in interior faces, discard them

            if len(points) == 0:
                return None, None
            depth = (points + camera_t)[:, -1]
            index_ray = index_ray[discriminator]
            pixel_ray = coords[index_ray]

            minval = np.min(depth)
            maxval = np.max(depth)
            depthmap = np.zeros([H, W])

            depthmap[pixel_ray[:, 0], pixel_ray[:, 1]] = 1.0 - (0.8 * (depth - minval) / (maxval - minval))
            depthmap *= 255
        return depthmap, pred_2d_joints_img_space

    def get_depth(self, np_image, padding):
        info = {}

        # STEP 3: Load the input image.
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image.copy())

        # STEP 4: Detect hand landmarks from the input image.
        detection_result = self.detector.detect(image)

        handedness_list = detection_result.handedness
        hand_landmarks_list = detection_result.hand_landmarks

        raw_image = image.numpy_view()
        H, W, C = raw_image.shape

        # HANDLANDMARKS CAN BE EMPTY, HANDLE THIS!
        if len(hand_landmarks_list) == 0:
            return None, None, None
        raw_image = raw_image[:, :, :3]

        padded_image = np.zeros((H * 2, W * 2, 3))
        padded_image[int(1 / 2 * H) : int(3 / 2 * H), int(1 / 2 * W) : int(3 / 2 * W)] = raw_image

        hand_landmarks_list, handedness_list = zip(
            *sorted(zip(hand_landmarks_list, handedness_list), key=lambda x: x[0][9].z, reverse=True)
        )

        padded_depthmap = np.zeros((H * 2, W * 2))
        mask = np.zeros((H, W))
        crop_boxes = []
        # bboxes = []
        groundtruth_2d_keypoints = []
        hands = []
        depth_failure = False
        crop_lens = []
        abs_boxes = []

        true_hand_category = {"Right": "right", "Left": "left"}
        for idx in range(len(hand_landmarks_list)):
            hand = true_hand_category[handedness_list[idx][0].category_name]
            hands.append(hand)
            hand_landmarks = hand_landmarks_list[idx]
            height, width, _ = raw_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]

            # x_min, x_max, y_min, y_max: extrema from mediapipe keypoint detection
            x_min = int(min(x_coordinates) * width)
            x_max = int(max(x_coordinates) * width)
            x_c = (x_min + x_max) // 2
            y_min = int(min(y_coordinates) * height)
            y_max = int(max(y_coordinates) * height)
            y_c = (y_min + y_max) // 2
            abs_boxes.append([x_min, x_max, y_min, y_max])

            # if x_max - x_min < 60 or y_max - y_min < 60:
            #    continue

            crop_len = (max(x_max - x_min, y_max - y_min) * 1.6) // 2 * 2

            # crop_x_min, crop_x_max, crop_y_min, crop_y_max: bounding box for mesh reconstruction
            crop_x_min = int(x_c - (crop_len / 2 - 1) + W / 2)
            crop_x_max = int(x_c + crop_len / 2 + W / 2)
            crop_y_min = int(y_c - (crop_len / 2 - 1) + H / 2)
            crop_y_max = int(y_c + crop_len / 2 + H / 2)

            cropped = padded_image[crop_y_min : crop_y_max + 1, crop_x_min : crop_x_max + 1]
            crop_boxes.append([crop_y_min, crop_y_max, crop_x_min, crop_x_max])
            crop_lens.append(crop_len)

            if hand == "left":
                cropped = cv2.flip(cropped, 1)

            if crop_len < 224:
                graphormer_input = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_CUBIC)
            else:
                graphormer_input = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA)
            scale = crop_len / 224
            cropped_depthmap, pred_2d_keypoints = self.run_inference(
                graphormer_input.astype(np.uint8), self._model, self.mano_model, self.mesh_sampler, scale, int(crop_len)
            )

            if cropped_depthmap is None:
                depth_failure = True
                break
            # keypoints_image_space = pred_2d_keypoints * (crop_y_max - crop_y_min + 1)/224
            groundtruth_2d_keypoints.append(pred_2d_keypoints)

            if hand == "left":
                cropped_depthmap = cv2.flip(cropped_depthmap, 1)
            resized_cropped_depthmap = cv2.resize(
                cropped_depthmap, (int(crop_len), int(crop_len)), interpolation=cv2.INTER_LINEAR
            )
            nonzero_y, nonzero_x = (resized_cropped_depthmap != 0).nonzero()

            if len(nonzero_y) == 0 or len(nonzero_x) == 0:
                depth_failure = True
                break
            padded_depthmap[crop_y_min + nonzero_y, crop_x_min + nonzero_x] = resized_cropped_depthmap[
                nonzero_y, nonzero_x
            ]

            # nonzero stands for nonzero value on the depth map
            # coordinates of nonzero depth pixels in original image space
            original_nonzero_x = crop_x_min + nonzero_x - int(W / 2)
            original_nonzero_y = crop_y_min + nonzero_y - int(H / 2)

            nonzerox_min = min(np.min(original_nonzero_x), x_min)
            nonzerox_max = max(np.max(original_nonzero_x), x_max)
            nonzeroy_min = min(np.min(original_nonzero_y), y_min)
            nonzeroy_max = max(np.max(original_nonzero_y), y_max)

            bbx_min, bbx_max, bby_min, bby_max = self.get_mask_bounding_box(
                (nonzerox_min, nonzerox_max, nonzeroy_min, nonzeroy_max), H, W, padding
            )
            mask[bby_min : bby_max + 1, bbx_min : bbx_max + 1] = 1.0
            # bboxes.append([int(bbx_min), int(bbx_max), int(bby_min), int(bby_max)])

        if depth_failure:
            return None, None, None

        depthmap = padded_depthmap[int(1 / 2 * H) : int(3 / 2 * H), int(1 / 2 * W) : int(3 / 2 * W)].astype(np.uint8)
        mask = (255.0 * mask).astype(np.uint8)
        info["groundtruth_2d_keypoints"] = groundtruth_2d_keypoints
        info["hands"] = hands
        info["crop_boxes"] = crop_boxes
        info["crop_lens"] = crop_lens
        info["abs_boxes"] = abs_boxes

        return depthmap, mask, info
