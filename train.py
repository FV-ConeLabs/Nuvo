import torch
import trimesh
import wandb
import os
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
import numpy as np

from network import Nuvo
from utils import (
    sample_points_on_mesh,
    sample_uv_points,
    sample_triangles,
    create_rgb_maps,
    set_all_seeds,
    normalize_mesh,
    create_wandb_object,
)
from loss import compute_loss


def main(config_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf = OmegaConf.load(config_path)
    set_all_seeds(conf.train.seed)
    model = Nuvo(**conf.model).to(device)
    sigma = nn.Parameter(torch.tensor(1.0, device=device))
    texture_map_res = int(256 * ((2 / conf.model.num_charts) ** 0.5))
    normal_maps = nn.Parameter(
        torch.zeros(
            conf.model.num_charts, texture_map_res, texture_map_res, 3, device=device
        )
    )

    # optimizers and schedulers
    T_max = conf.train.epochs * conf.train.iters
    optimizer_nuvo = torch.optim.Adam(model.parameters(), lr=conf.optimizer.nuvo_lr)
    scheduler_nuvo = CosineAnnealingLR(optimizer_nuvo, T_max=T_max)
    optimizer_sigma = torch.optim.Adam([sigma], lr=conf.optimizer.sigma_lr)
    scheduler_sigma = CosineAnnealingLR(optimizer_sigma, T_max=T_max)
    optimizer_normal_maps = torch.optim.Adam(
        [normal_maps], lr=conf.optimizer.normal_grids_lr
    )
    scheduler_normal_maps = CosineAnnealingLR(optimizer_normal_maps, T_max=T_max)

    start_iteration = 0
    if "ckpt" in conf and conf.ckpt:
        print(f"Loading checkpoint from {conf.ckpt}...")
        ckpt = torch.load(conf.ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        sigma.data = ckpt["sigma"]
        optimizer_nuvo.load_state_dict(ckpt["optimizer_nuvo_state_dict"])
        scheduler_nuvo.load_state_dict(ckpt["scheduler_nuvo_state_dict"])
        optimizer_sigma.load_state_dict(ckpt["optimizer_sigma_state_dict"])
        scheduler_sigma.load_state_dict(ckpt["scheduler_sigma_state_dict"])
        start_iteration = ckpt.get("iteration", 0)

    # optional wandb logging
    if conf.train.use_wandb:
        wandb.config = OmegaConf.to_container(conf, resolve=True)
        wandb.init(project="Nuvo", name=conf.train.name)

    # load mesh
    mesh = trimesh.load_mesh(conf.train.mesh_path)
    mesh = normalize_mesh(mesh)
    # print(f"Mesh has {len(mesh.vertices)} vertices. Subdividing...")
    # mesh = mesh.subdivide()
    # print(f"After subdivision, mesh has {len(mesh.vertices)} vertices.")

    # train loop
    for epoch in range(conf.train.epochs):
        for i in tqdm(range(start_iteration, conf.train.iters), desc=f"Epoch {epoch+1}/{conf.train.epochs}"):
            optimizer_nuvo.zero_grad()
            optimizer_sigma.zero_grad()
            optimizer_normal_maps.zero_grad()

            # sample points on mesh and UV points
            points, normals = sample_points_on_mesh(mesh, conf.train.G_num)
            uvs = sample_uv_points(conf.train.T_num)
            points = torch.tensor(points, dtype=torch.float32, device=device)
            normals = torch.tensor(normals, dtype=torch.float32, device=device)
            uvs = torch.tensor(uvs, dtype=torch.float32, device=device)

            # Sample triangles for consistency loss
            tri_verts = None
            if conf.loss.get("triangle_consistency", 0.0) > 0:
                tri_verts = sample_triangles(mesh, conf.train.triangle_num, device)

            # compute losses
            loss_dict = compute_loss(
                conf,
                points,
                normals,
                uvs,
                model,
                sigma,
                normal_maps,
                tri_verts,
            )

            loss_dict["loss_combined"].backward()
            optimizer_nuvo.step()
            optimizer_sigma.step()
            optimizer_normal_maps.step()
            scheduler_nuvo.step()
            scheduler_sigma.step()
            scheduler_normal_maps.step()

            # logging
            if conf.train.use_wandb:
                wandb.log({k: v.item() for k, v in loss_dict.items()})
                wandb.log({"sigma": sigma.item()})

                if (i + 1) % conf.train.texture_map_save_interval == 0:
                    val_normal_maps = normal_maps.permute(0, 3, 1, 2)
                    val_normal_maps = make_grid(val_normal_maps, nrow=2)
                    wandb.log({"normal_maps": [wandb.Image(val_normal_maps)]})
                    val_wandb_object, new_mesh = create_wandb_object(
                        mesh, device, model
                    )
                    wandb.log(
                        {
                            "segmented mesh w.r.t chart distribution": val_wandb_object,
                        }
                    )

                    mesh_path = os.path.join(wandb.run.dir, f"mesh_{epoch}_{i}.obj")
                    new_mesh.export(mesh_path)
            else:
                if i % 1000 == 0:
                    print(
                        f"Epoch: {epoch}, Iter: {i}, Total Loss: {loss_dict['loss_combined'].item()}"
                    )
            
            if (i + 1) % conf.train.iters == 0:
                ckpt = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_nuvo_state_dict": optimizer_nuvo.state_dict(),
                    "scheduler_nuvo_state_dict": scheduler_nuvo.state_dict(),
                    "optimizer_sigma_state_dict": optimizer_sigma.state_dict(),
                    "scheduler_sigma_state_dict": scheduler_sigma.state_dict(),
                    "sigma": sigma.data,
                    "epoch": epoch,
                    "iteration": i,
                }
                os.makedirs("output", exist_ok=True)
                torch.save(ckpt, f"output/checkpoint_{epoch}_{i}.ckpt")
        
    # --- START: NEW FINAL MESH EXPORT LOGIC ---
    print("Training finished. Starting final UV generation for all vertices...")
    # Use the same mesh that was used for training
    final_export_mesh = mesh

    vertices = torch.tensor(final_export_mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(final_export_mesh.faces, dtype=torch.int64, device=device)
    num_vertices = vertices.shape[0]
    num_faces = faces.shape[0]
    inference_batch_size = 10000

    # 1. Get chart probabilities for all vertices
    print(f"Processing {num_vertices} vertices to get chart probabilities...")
    all_chart_probs = []
    with torch.no_grad():
        model.eval()
        for i in tqdm(range(0, num_vertices, inference_batch_size), desc="Vertex Chart Probs"):
            batch_vertices = vertices[i:i + inference_batch_size]
            batch_chart_probs = model.chart_assignment_mlp(batch_vertices)
            all_chart_probs.append(batch_chart_probs)
    chart_probs = torch.cat(all_chart_probs, dim=0)

    # 2. Determine a chart for each FACE (triangle) using majority vote
    print(f"Assigning charts to {num_faces} faces by majority vote...")
    face_vert_probs = chart_probs[faces]            # Shape: [num_faces, 3, num_charts]
    summed_face_probs = torch.sum(face_vert_probs, dim=1) # Sum probs across vertices
    face_chart_indices = torch.argmax(summed_face_probs, dim=1) # Shape: [num_faces]

    # 3. Create a new "unwelded" mesh to handle UV seams
    print("Unwelding mesh and calculating per-corner UVs...")
    final_vertices_tensor = vertices[faces].reshape(-1, 3)
    final_faces_tensor = torch.arange(len(final_vertices_tensor), device=device).view(-1, 3)

    # 4. Calculate UVs for each new vertex (face corner) based on its face's chart
    final_uvs_tensor = torch.zeros(final_vertices_tensor.shape[0], 2, device=device)
    with torch.no_grad():
        for chart_idx in tqdm(range(conf.model.num_charts), desc="Calculating UVs per chart"):
            mask = (face_chart_indices == chart_idx)
            if not mask.any():
                continue

            all_corner_indices_in_chart = final_faces_tensor[mask].reshape(-1)
            all_corners_to_process = final_vertices_tensor[all_corner_indices_in_chart]

            num_corners_in_chart = len(all_corners_to_process)

            for i in range(0, num_corners_in_chart, inference_batch_size):
                batch_corner_indices = all_corner_indices_in_chart[i:i + inference_batch_size]
                batch_corners = all_corners_to_process[i:i + inference_batch_size]

                chart_ids_for_mlp = torch.full((len(batch_corners),), fill_value=chart_idx, device=device, dtype=torch.long)

                uv_coords = model.texture_coordinate_mlp(batch_corners, chart_ids_for_mlp)

                final_uvs_tensor[batch_corner_indices] = uv_coords

    # 5. Tile the UVs into a single atlas layout
    print("Tiling UVs into atlas layout...")
    uvs_np = final_uvs_tensor.cpu().numpy()
    vertex_chart_indices = torch.repeat_interleave(face_chart_indices, 3).cpu().numpy()

    uvs_np[:, 0] = (uvs_np[:, 0] + vertex_chart_indices) / conf.model.num_charts
    uvs_np = np.clip(uvs_np, 0.0, 1.0) # Ensure UVs are in the valid [0,1] range

    # 6. Create and export the final Trimesh object
    print("Creating final Trimesh object...")
    tex_visuals = trimesh.visual.TextureVisuals(uv=uvs_np)
    final_mesh = trimesh.Trimesh(vertices=final_vertices_tensor.cpu().numpy(),
                                   faces=final_faces_tensor.cpu().numpy(),
                                   visual=tex_visuals,
                                   process=False) # Disable processing to keep unwelded vertices

    export_path = "output/final_mesh_consistent.obj"
    os.makedirs("output", exist_ok=True)
    print(f"Exporting final mesh with consistent UVs to {export_path}...")
    final_mesh.export(export_path)
    print("Export complete!")
    # --- END: NEW FINAL MESH EXPORT LOGIC ---


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, default="configs/nefertiti.yaml")
    args = args.parse_args()

    main(args.config)