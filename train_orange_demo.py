import sys

sys.path.append("gaussian-splatting")

import argparse
import cv2
import torch
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import random
import torchvision.transforms as T
# Gaussian splatting dependencies
from scene.gaussian_model import GaussianModel
from gaussian_renderer import GaussianModel
from utils.system_utils import searchForMaxIteration
from mpm_solver_warp.engine_utils import *
from pytorch_msssim import ssim
import Canny_Edge_Detection as CED

# Particle filling dependencies
from particle_filling.filling import *

# Utils
from utils.decode_param import *
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import *
import torchvision.transforms as transforms
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from cross_section import *
from new_sds_demo import *

CE_MODEL_VERTICAL = "./local_models/controlnet-canny"

SD_MODEL_VERTICAL="./local_models/sd2-depth"
SD_MODEL_HORIZONTAL="./local_models/sd2-depth"

class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def save_img(rendering, path, frame, f_prefix=""):
    cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    assert args.output_path is not None
    cv2_img *= 255
    path = os.path.join(path, f"{f_prefix}{frame}.png".rjust(8, "0"))
    cv2.imwrite(
        path,
        cv2_img,
    )
    return path


def load_checkpoint(model_path, iteration=-1, gs_path=None):
    if gs_path:
        checkpt_path = gs_path
        print("using ", gs_path)
    # sh_degree=0, if you use a 3D asset without spherical harmonics
    from plyfile import PlyData
    plydata = PlyData.read(checkpt_path)
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))

    sh_degree = 0
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply_zero_sh(checkpt_path)
    return gaussians


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--physics_config", type=str, required=True)
    parser.add_argument("--guidance_config", type=str, default="./config/guidance/ms_guidance.yaml")
    parser.add_argument("--white_bg", type=bool, default=True)
    parser.add_argument("--output_ply", action="store_true")
    parser.add_argument("--output_h5", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--gs_path", type=str, default=None)
    parser.add_argument("--gs_ori_path", type=str, default=None)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--model",type=str,default="local")
    parser.add_argument("--input_path",type=str,default=None,required=False)

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        AssertionError("Model path does not exist!")
    if not os.path.exists(args.physics_config):
        AssertionError("Scene config does not exist!")
    if not os.path.exists(args.guidance_config):
        AssertionError("Scene config does not exist!")
    if args.output_path is not None and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    train = args.train

    # load scene config
    print("Loading scene config...")
    (
        material_params,
        bc_params,
        time_params,
        preprocessing_params,
        camera_params,
    ) = decode_param_json(args.physics_config)

    # load gaussians
    print("Loading gaussians...")
    model_path = args.model_path
    gaussians = load_checkpoint(model_path, gs_path=args.gs_path)
    gaussians_ori = load_checkpoint(model_path, gs_path=args.gs_ori_path)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background_b = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    )
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    )
    params = load_params_from_gs(gaussians, pipeline)
    init_opacity = params["opacity"]
    max_value = init_opacity.max()
    max_value = 10000
    # Detach the tensor from the computation graph, modify it, and then reattach it
    with torch.no_grad():
        gaussians._opacity.copy_(init_opacity.clone().detach().fill_(max_value))

    # init the scene
    print("Initializing scene and pre-processing...")


    def create_3d_grid(gaussians, grid_size):
        xyz = gaussians.get_xyz  # Shape (N, 3)
        device = xyz.device
        min_coords = xyz.min(dim=0)[0]
        max_coords = xyz.max(dim=0)[0]
        cell_dimensions = (max_coords - min_coords) / torch.tensor(grid_size, device=device)
        grid = {}
        for idx in tqdm(range(xyz.size(0)), desc="Creating 3D Grid"):
            cell_coords = ((xyz[idx] - min_coords) / cell_dimensions).floor().long()
            cell_key = tuple(cell_coords.tolist())
            if cell_key not in grid:
                grid[cell_key] = []
            grid[cell_key].append(idx)
        return grid


    def smooth_gaussians_in_grid(gaussians, grid):
        scales = gaussians.get_scaling
        rotations = gaussians.get_rotation
        features = gaussians.get_features
        smoothed_scales = torch.zeros_like(scales)
        smoothed_rotations = torch.zeros_like(rotations)
        smoothed_features = torch.zeros_like(features)
        counts = torch.zeros(len(scales), dtype=torch.int)
        for cell_key, indices in tqdm(grid.items(), desc="Smoothing Gaussians in Grid"):
            if len(indices) > 0:
                cell_features = features[indices]
                avg_features = torch.mean(cell_features, dim=0).squeeze(0)
                smoothed_features[indices] = avg_features
                counts[indices] += 1
        gaussians._features_dc.copy_(smoothed_features.clone().detach())


    def preprocess_particles(gaussians, pipeline, preprocessing_params, args):
        params = load_params_from_gs(gaussians, pipeline)
        init_pos = params["pos"]
        init_cov = params["cov3D_precomp"]
        init_screen_points = params["screen_points"]
        init_opacity = params["opacity"]
        init_shs = params["shs"]

        if args.debug:
            log_dir = "./log"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            particle_position_tensor_to_ply(init_pos, os.path.join(log_dir, "init_particles.ply"))

        transformed_pos, scale_origin, original_mean_pos = transform2origin(init_pos)
        transformed_pos = shift2center111(transformed_pos)

        init_cov = apply_cov_rotations(init_cov, rotation_matrices)
        init_cov = scale_origin * scale_origin * init_cov

        if args.debug:
            particle_position_tensor_to_ply(transformed_pos, os.path.join(log_dir, "transformed_particles.ply"))

        device = "cuda:0"
        mpm_init_pos = transformed_pos.to(device=device)
        mpm_init_cov = init_cov
        return init_shs, init_opacity, mpm_init_pos, mpm_init_cov, scale_origin, original_mean_pos, init_screen_points


    filling_params = preprocessing_params["particle_filling"]

    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    )

    mpm_space_viewpoint_center = (
        torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    )
    mpm_space_vertical_upward_axis = (
        torch.tensor(camera_params["mpm_space_vertical_upward_axis"])
        .reshape((1, 3))
        .cuda()
    )


    class TrainingArgs:
        def __init__(self):
            self.position_lr_init = 0.001
            self.position_lr_final = 0.0002
            self.position_lr_delay_mult = 0.02
            self.position_lr_max_steps = 600
            self.feature_lr = 0.001
            self.opacity_lr = 0.01
            self.scaling_lr = 0.001
            self.rotation_lr = 0.01
            self.percent_dense = 0.01
            self.density_start_iter = 0
            self.density_end_iter = 3000
            self.densification_interval = 50
            self.opacity_reset_interval = 700
            self.densify_grad_threshold = 0.01


    training_args = TrainingArgs()
    gaussians.spatial_lr_scale = 0.1
    gaussians.training_setup(training_args)

    transform = transforms.ToTensor()
    device = "cuda:0"
    view_count = 180
    epochs = 400
    prev_loss = 0
    track_loss = 0
    steps_per_c = 3

    pos = gaussians.get_xyz


    def training_step(gaussians, loss, grad_update_mask=None, viewspace_point_tensor=None, visibility_filter=None):
        for group in gaussians.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    if grad_update_mask != None:
                        param.grad[~grad_update_mask] = 0
                    if group['name'] == 'opacity':
                        param.grad[:] = 0
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
        gaussians.optimizer.step()
        with torch.no_grad():
            gaussians._opacity.copy_(init_opacity.clone().detach().fill_(max_value))
            gaussians._scaling.clamp_(max=-16)
        gaussians.optimizer.zero_grad()


    def density_and_prune(e=0):
        if e > 1 and e % 10 == 0:
            print("before densifying")
            print(gaussians.get_xyz.shape)
            grads = gaussians.xyz_gradient_accum / gaussians.denom
            grads[grads.isnan()] = 0.0
            mean_grads = torch.mean(grads)
            max_grads = torch.max(grads)
            min_grads = torch.min(grads)
            print(f"Gradient Statistics:")
            print(f"Mean: {mean_grads.item()}")
            print(f"Max: {max_grads.item()}")
            print(f"Min: {min_grads.item()}")
            gaussians.densify_and_prune(0.0002, min_opacity=0.0000001, extent=4, max_screen_size=None)
            print("after densifying")
            print(gaussians.get_xyz.shape)

        if e > 0 and e % 101 == 0:
            with torch.no_grad():
                grid = create_3d_grid(gaussians=gaussians, grid_size=(512, 512, 512))
                smooth_gaussians_in_grid(gaussians, grid)


    def get_ssim_loss(rendering, ground_truth):
        rendering = rendering.unsqueeze(0)
        ground_truth = ground_truth.unsqueeze(0)

        return 1 - ssim(
            rendering,
            ground_truth,
            data_range=1,
            size_average=True
        )


    with torch.no_grad():
        gaussians._scaling.clamp_(max=-16)

    now_pic_v=0
    now_pic_h=0
    if args.model=="CED":
        # [Original Code - commented out for memory optimization comparison]
        # controlnet = ControlNetModel.from_pretrained(CE_MODEL_VERTICAL, torch_dtype=torch.float16)
        #
        # pipe_d_v = StableDiffusionDepth2ImgPipeline.from_pretrained(SD_MODEL_VERTICAL).to("cuda:0")
        # pipe_d_h = StableDiffusionDepth2ImgPipeline.from_pretrained(SD_MODEL_HORIZONTAL).to("cuda:0")
        #
        # pipe_ce = StableDiffusionControlNetPipeline.from_pretrained(
        #     "runwayml/stable-diffusion-v1-5",
        #     controlnet=controlnet,
        #     torch_dtype=torch.float16
        # ).to("cuda:0")

        # [Change 1] Merge the two identical depth pipelines into one shared instance.
        # Vertical/horizontal slicing still differs by input rendering, depth map and prompt,
        # so sharing the same weights does not change the intended training logic.
        controlnet = ControlNetModel.from_pretrained(CE_MODEL_VERTICAL, torch_dtype=torch.float16)

        # [Original Code - commented out for compatibility comparison]
        # pipe_d = StableDiffusionDepth2ImgPipeline.from_pretrained(
        #     SD_MODEL_VERTICAL,
        #     torch_dtype=torch.float16,
        # ).to("cuda:0")
        # pipe_d.enable_attention_slicing()
        # pipe_d.enable_vae_slicing()

        # [Original Code - commented out for ref-image recovery comparison]
        # pipe_d = StableDiffusionDepth2ImgPipeline.from_pretrained(
        #     SD_MODEL_VERTICAL,
        #     torch_dtype=torch.float16,
        # ).to("cuda:0")
        # if hasattr(pipe_d, "enable_attention_slicing"):
        #     pipe_d.enable_attention_slicing()
        # if hasattr(pipe_d, "enable_vae_slicing"):
        #     pipe_d.enable_vae_slicing()

        # [Change 1.2] Keep the shared depth pipeline, but restore its default precision.
        # This is intended to avoid fp16 instability in the SDS/depth/VAE chain that can produce black ref images.
        pipe_d = StableDiffusionDepth2ImgPipeline.from_pretrained(
            SD_MODEL_VERTICAL,
        ).to("cuda:0")
        if hasattr(pipe_d, "enable_attention_slicing"):
            pipe_d.enable_attention_slicing()
        if hasattr(pipe_d, "enable_vae_slicing"):
            pipe_d.enable_vae_slicing()

        # [Change 2] Keep only one ControlNet pipeline resident on GPU and enable slicing.
        # [Original Code - commented out for compatibility comparison]
        # pipe_ce = StableDiffusionControlNetPipeline.from_pretrained(
        #     "runwayml/stable-diffusion-v1-5",
        #     controlnet=controlnet,
        #     torch_dtype=torch.float16
        # ).to("cuda:0")
        # pipe_ce.enable_attention_slicing()
        # pipe_ce.enable_vae_slicing()

        # [Change 2.1] Guard optional memory-saving APIs for older diffusers versions.
        pipe_ce = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to("cuda:0")
        if hasattr(pipe_ce, "enable_attention_slicing"):
            pipe_ce.enable_attention_slicing()
        if hasattr(pipe_ce, "enable_vae_slicing"):
            pipe_ce.enable_vae_slicing()

    for j in range(3000):
        density_and_prune(j)
        print(f"Starting iteration {j}")

        # ==========================================
        # 循环 1: Vertical views
        # ==========================================
        for i in range(30):
            print(f"Starting v{i}/30")
            el = 0
            torch.cuda.empty_cache()
            init_shs, init_opacity, mpm_init_pos, mpm_init_cov, scale_origin, original_mean_pos, init_screen_points = preprocess_particles(
                gaussians, pipeline, preprocessing_params, args)
            shs_render = init_shs
            opacity_render = init_opacity
            (
                viewpoint_center_worldspace,
                observant_coordinates,
            ) = get_center_view_worldspace_and_observant_coordinate(
                mpm_space_viewpoint_center,
                mpm_space_vertical_upward_axis,
                rotation_matrices,
                scale_origin,
                original_mean_pos,
            )
            cur_camera, raw_camera = get_camera_view(
                model_path,
                default_camera_index=-1,
                center_view_world_space=viewpoint_center_worldspace,
                observant_coordinates=observant_coordinates,
                show_hint=False,
                init_azimuthm=12 * i,
                init_elevation=el,
                init_radius=camera_params["init_radius"],
                move_camera=False,
                current_frame=0,
                delta_a=None, delta_e=None, delta_r=None
            )
            torch.cuda.empty_cache()
            pos = mpm_init_pos
            cov3D = mpm_init_cov
            rot = None

            cov3D = cov3D / (scale_origin * scale_origin)
            cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
            opacity = opacity_render
            shs = shs_render

            plane = generate_plane(raw_camera, filling_params["boundary"])
            thickness = 0.006
            mask, _ = plane_filter(plane, pos, raw_camera, surf_dis=thickness, include_double=True)
            pos = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(pos), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )
            pos_cs = pos[mask]
            shs_cs = shs[mask]
            cov3D_cs = cov3D[mask]
            opacity_cs = opacity[mask]
            init_screen_points_cs = init_screen_points[mask]
            rasterize = initialize_resterize(
                cur_camera, gaussians, pipeline, background, image_height=512, image_width=512
            )
            colors_precomp_cs = convert_SH(shs_cs, cur_camera, gaussians, pos_cs, None)

            rendering, raddi = rasterize(
                means3D=pos_cs,
                means2D=init_screen_points_cs,
                shs=None,
                colors_precomp=colors_precomp_cs,
                opacities=opacity_cs,
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D_cs
            )

            target_size = (384, 384)
            depth_map = rendering
            depth_map = F.interpolate(
                rendering.unsqueeze(0),
                size=target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            # [Original Code - commented out for memory optimization comparison]
            # pipe=pipe_d_v
            # depth_map = depth_map[None, :, :, :]
            # depth_map = depth_map.to("cuda:0")
            # depth_map = pipe.depth_estimator(depth_map).predicted_depth

            # [Original Code - commented out for dtype compatibility comparison]
            # pipe = pipe_d
            # depth_map = depth_map[None, :, :, :]
            # depth_map = depth_map.to(pipe.device)
            # with torch.no_grad():
            #     depth_map = pipe.depth_estimator(depth_map).predicted_depth

            # [Change 3.1] Match the depth estimator input dtype to the half-precision pipeline.
            pipe = pipe_d
            depth_map = depth_map[None, :, :, :]
            depth_map = depth_map.to(device=pipe.device, dtype=pipe.dtype)
            with torch.no_grad():
                depth_map = pipe.depth_estimator(depth_map).predicted_depth
            depth_map_tensor_resized = F.interpolate(depth_map.unsqueeze(0), size=target_size, mode='bilinear',
                                                     align_corners=False)
            depth_map_tensor_resized = depth_map_tensor_resized.squeeze(0)

            save_img(rendering, args.output_path, 0, f"v{i}_init_")

            if j % 30 == 0:
                cur_img = Image.open(os.path.join(args.output_path, f"v{i}_init_0.png"))
                if args.model=="local":
                    try:
                        ref = Image.open(os.path.join(args.input_path, f"v{now_pic_v}.png"))
                    except FileNotFoundError:
                        now_pic_v = 0
                        ref = Image.open(os.path.join(args.input_path, f"v{now_pic_v}.png"))
                    now_pic_v += 1
                if args.model=="CED":
                    if j>=300:
                        canny_condition_img_path = CED.get_canny_edges(
                            os.path.join(args.output_path, f"v{i}_init_0.png"))
                        ref = one_step_c_orange(cur_img, canny_condition_img_path, 30 - j // 100, pipe_ce, "vertical")
                        ref.save(os.path.join(args.output_path, f"h{i}_ref.png"))
                    else:
                        cur_img = Image.open(os.path.join(args.output_path, f"v{i}_init_0.png"))
                        # [Original Code - commented out for memory optimization comparison]
                        # ref = one_step_sds_orange(cur_img, depth_map_tensor_resized, 30 - j // 100, pipe, "vertical")

                        # [Change 4] Use a small fixed latent optimization step count to reduce peak memory.
                        ref = one_step_sds_orange(cur_img, depth_map_tensor_resized, 4, pipe, "vertical")
                        ref.save(os.path.join(args.output_path, f"v{i}_ref.png"))
                ref.save(os.path.join(args.output_path, f"v{i}_ref.png"))
            else:
                ref = Image.open(os.path.join(args.output_path, f"v{i}_ref.png"))

            ground_truth_tensor = transform(ref).to(device)
            if ground_truth_tensor.shape[0] == 4:
                ground_truth_tensor = ground_truth_tensor[:3, :, :]
            if ground_truth_tensor.shape[1:] != target_size:
                ground_truth_tensor = F.interpolate(
                    ground_truth_tensor.unsqueeze(0),
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            tmp_rendering=rendering
            tmp_rendering.unsqueeze_(0)
            tmp_rendering = F.interpolate(
                tmp_rendering,
                size=ground_truth_tensor.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            tmp_rendering.squeeze_(0)
            total_loss = 0.7 * get_ssim_loss(tmp_rendering, ground_truth_tensor)
            total_loss += 0.3 * torch.nn.functional.mse_loss(tmp_rendering, ground_truth_tensor)
            total_loss.backward()
            output_radii = torch.zeros(pos.shape[0], dtype=torch.int32).to(device)
            output_radii[mask] = raddi
            visibility_filter = output_radii > 0
            training_step(gaussians, total_loss, mask, init_screen_points, visibility_filter)

        # ==========================================
        # 循环 2: Horizontal views
        # ==========================================
        torch.cuda.empty_cache()
        init_shs, init_opacity, mpm_init_pos, mpm_init_cov, scale_origin, original_mean_pos, init_screen_points = preprocess_particles(
            gaussians, pipeline, preprocessing_params, args)
        shs_render = init_shs
        opacity_render = init_opacity
        (
            viewpoint_center_worldspace,
            observant_coordinates,
        ) = get_center_view_worldspace_and_observant_coordinate(
            mpm_space_viewpoint_center,
            mpm_space_vertical_upward_axis,
            rotation_matrices,
            scale_origin,
            original_mean_pos,
        )
        cur_camera, raw_camera = get_camera_view(
            model_path,
            default_camera_index=-1,
            center_view_world_space=viewpoint_center_worldspace,
            observant_coordinates=observant_coordinates,
            show_hint=False,
            init_azimuthm=0,
            init_elevation=90,
            init_radius=camera_params["init_radius"],
            move_camera=False,
            current_frame=0,
            delta_a=None, delta_e=None, delta_r=None
        )
        torch.cuda.empty_cache()
        pos = mpm_init_pos
        cov3D = mpm_init_cov
        rot = None
        steps = 70
        _, _, centers, avg_dis = interpolate_along_camera_direction(raw_camera, pos, steps)
        avg_dis = avg_dis.item()

        for i, c in enumerate(centers[10:60]):
            print(f"Starting h{i}/{len(centers)}")
            init_shs, init_opacity, mpm_init_pos, mpm_init_cov, scale_origin, original_mean_pos, init_screen_points = preprocess_particles(
                gaussians, pipeline, preprocessing_params, args)
            shs_render = init_shs
            opacity_render = init_opacity
            torch.cuda.empty_cache()
            pos = mpm_init_pos
            cov3D = mpm_init_cov
            rot = None
            opacity = opacity_render
            shs = shs_render
            cov3D = cov3D / (scale_origin * scale_origin)
            plane = generate_plane_center(raw_camera, c)
            mask, mask_suf = plane_filter(plane, pos, raw_camera, surf_dis=avg_dis / 2, include_double=True)
            pos = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(pos), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )
            cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
            pos_cs = pos[mask_suf]
            shs_cs = shs[mask_suf]
            cov3D_cs = cov3D[mask_suf]
            opacity_cs = opacity[mask_suf]
            init_screen_points_cs = init_screen_points[mask_suf]
            colors_precomp_cs = convert_SH(shs_cs, cur_camera, gaussians, pos_cs, None)
            rasterize = initialize_resterize(
                cur_camera, gaussians, pipeline, background, image_height=512, image_width=512
            )

            rendering, raddi = rasterize(
                means3D=pos_cs,
                means2D=init_screen_points_cs,
                shs=None,
                colors_precomp=colors_precomp_cs,
                opacities=opacity_cs,
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D_cs
            )

            target_size = (384, 384)
            depth_map = rendering
            depth_map= F.interpolate(
                rendering.unsqueeze(0),
                size=target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            # [Original Code - commented out for memory optimization comparison]
            # pipe=pipe_d_h
            # depth_map = depth_map[None,:,:,:]
            # depth_map = depth_map.to("cuda:1")
            # depth_map = pipe.depth_estimator(depth_map).predicted_depth

            # [Original Code - commented out for dtype compatibility comparison]
            # pipe = pipe_d
            # depth_map = depth_map[None, :, :, :]
            # depth_map = depth_map.to(pipe.device)
            # with torch.no_grad():
            #     depth_map = pipe.depth_estimator(depth_map).predicted_depth

            # [Change 5.1] Match the horizontal depth estimator input dtype to the pipeline as well.
            pipe = pipe_d
            depth_map = depth_map[None, :, :, :]
            depth_map = depth_map.to(device=pipe.device, dtype=pipe.dtype)
            with torch.no_grad():
                depth_map = pipe.depth_estimator(depth_map).predicted_depth
            target_size = (512, 512)
            depth_map_tensor_resized = F.interpolate(depth_map.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
            depth_map_tensor_resized = depth_map_tensor_resized.squeeze(0)
            save_img(rendering, args.output_path, 0, f"h{i}_init_")

            if j % 30 == 0:
                cur_img = Image.open(os.path.join(args.output_path, f"h{i}_init_0.png"))
                if args.model == "local":
                    try:
                        ref = Image.open(os.path.join(args.input_path, f"h{now_pic_h}.png"))
                    except FileNotFoundError:
                        now_pic_h = 0
                        ref = Image.open(os.path.join(args.input_path, f"h{now_pic_h}.png"))
                        now_pic_h = now_pic_h + 1
                    ref.save(os.path.join(args.output_path, f"h{i}_ref.png"))
                if args.model == "CED":
                    if j>=300:
                        canny_condition_img_path = CED.get_canny_edges(
                            os.path.join(args.output_path, f"h{i}_init_0.png"))
                        # [Original Code - commented out for memory optimization comparison]
                        # ref = one_step_c_orange(cur_img, canny_condition_img_path, 30 - j // 100, pipe_ce, "horizontal")

                        # [Change 6] Reduce ControlNet latent optimization steps to improve stability on 8GB VRAM.
                        ref = one_step_c_orange(cur_img, canny_condition_img_path, 4, pipe_ce, "horizontal")
                        ref.save(os.path.join(args.output_path, f"h{i}_ref.png"))
                    else :

                        cur_img = Image.open(os.path.join(args.output_path, f"h{i}_init_0.png"))
                        # [Original Code - commented out for memory optimization comparison]
                        # ref = one_step_sds_orange(cur_img, depth_map_tensor_resized, 30 - j // 100, pipe, "horizontal")

                        # [Change 7] Reduce SDS latent optimization steps for horizontal slices as well.
                        ref = one_step_sds_orange(cur_img, depth_map_tensor_resized, 4, pipe, "horizontal")
                        ref.save(os.path.join(args.output_path, f"h{i}_ref.png"))
            else:
                ref = Image.open(os.path.join(args.output_path, f"h{i}_ref.png"))

            ground_truth_tensor = transform(ref).to(device)
            if ground_truth_tensor.shape[0] == 4:
                ground_truth_tensor = ground_truth_tensor[:3, :, :]
            if ground_truth_tensor.shape[1:] != target_size:
                ground_truth_tensor = F.interpolate(
                    ground_truth_tensor.unsqueeze(0),
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            tmp_rendering = rendering
            tmp_rendering.unsqueeze_(0)
            tmp_rendering = F.interpolate(
                tmp_rendering,
                size=ground_truth_tensor.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            tmp_rendering.squeeze_(0)
            total_loss = 0.7 * get_ssim_loss(tmp_rendering, ground_truth_tensor)
            total_loss += 0.3 * torch.nn.functional.mse_loss(tmp_rendering, ground_truth_tensor)
            total_loss.backward()
            output_radii = torch.zeros(pos.shape[0], dtype=torch.int32).to(device)
            output_radii[mask_suf] = raddi
            visibility_filter = output_radii > 0
            training_step(gaussians, total_loss, mask_suf, init_screen_points, visibility_filter)

        # ==========================================
        # 循环 3: Original views (gaussians_ori)
        # ==========================================
        for ttt in range(30):
            print(f"Starting ori{ttt}/30")
            init_shs, init_opacity, mpm_init_pos, mpm_init_cov, scale_origin, original_mean_pos, init_screen_points = preprocess_particles(
                gaussians_ori, pipeline, preprocessing_params, args)
            shs_render = init_shs
            opacity_render = init_opacity
            torch.cuda.empty_cache()
            (
                viewpoint_center_worldspace,
                observant_coordinates,
            ) = get_center_view_worldspace_and_observant_coordinate(
                mpm_space_viewpoint_center,
                mpm_space_vertical_upward_axis,
                rotation_matrices,
                scale_origin,
                original_mean_pos,
            )
            cur_camera, raw_camera = get_camera_view(
                model_path,
                default_camera_index=-1,
                center_view_world_space=viewpoint_center_worldspace,
                observant_coordinates=observant_coordinates,
                show_hint=False,
                init_azimuthm=random.randint(0, 360),
                init_elevation=random.randint(-90, 90),
                init_radius=camera_params["init_radius"],
                move_camera=False,
                current_frame=0,
                delta_a=None, delta_e=None, delta_r=None
            )
            pos = mpm_init_pos
            cov3D = mpm_init_cov
            rot = None
            opacity = opacity_render
            shs = shs_render
            cov3D = cov3D / (scale_origin * scale_origin)
            pos = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(pos), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )
            cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
            colors_precomp = convert_SH(shs, cur_camera, gaussians_ori, pos, None)
            rasterize = initialize_resterize(
                cur_camera, gaussians_ori, pipeline, background, image_height=512, image_width=512
            )

            rendering_ori, radii_ori = rasterize(
                means3D=pos,
                means2D=init_screen_points,
                shs=None,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D
            )

            for p in range(1):
                init_shs, init_opacity, mpm_init_pos, mpm_init_cov, scale_origin, original_mean_pos, init_screen_points = preprocess_particles(
                    gaussians, pipeline, preprocessing_params, args)
                shs_render = init_shs
                opacity_render = init_opacity
                torch.cuda.empty_cache()
                pos = mpm_init_pos
                cov3D = mpm_init_cov
                opacity = opacity_render
                shs = shs_render
                cov3D = cov3D / (scale_origin * scale_origin)
                pos = apply_inverse_rotations(
                    undotransform2origin(
                        undoshift2center111(pos), scale_origin, original_mean_pos
                    ),
                    rotation_matrices,
                )
                colors_precomp = convert_SH(shs, cur_camera, gaussians, pos, None)
                rasterize = initialize_resterize(
                    cur_camera, gaussians, pipeline, background, image_height=512, image_width=512
                )

                rendering, radii = rasterize(
                    means3D=pos,
                    means2D=init_screen_points,
                    shs=None,
                    colors_precomp=colors_precomp,
                    opacities=opacity,
                    scales=None,
                    rotations=None,
                    cov3D_precomp=cov3D
                )

                save_img(rendering, args.output_path, 0, f"o{ttt}_init_")
                ground_truth_tensor = rendering_ori.detach()
                total_loss = 0.6 * get_ssim_loss(rendering, ground_truth_tensor)
                total_loss += 0.4 * torch.nn.functional.mse_loss(rendering, ground_truth_tensor)
                total_loss.backward()
                visibility_filter = radii > 0
                training_step(gaussians, total_loss, None, init_screen_points, visibility_filter)

        if j > 1 and j % 10 == 0:
            print("Saving epoch")
            gaussians.save_ply(os.path.join(args.output_path, f"orange_demo_epoch_{j}.ply"))
