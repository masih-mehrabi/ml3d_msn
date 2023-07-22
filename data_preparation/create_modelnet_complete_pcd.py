import trimesh
import open3d as o3d
import os
from open3d.geometry import PointCloud
from open3d.utility import Vector3dVector

data_path = "/content/ml3d_msn/data/ModelNet10/ModelNet10"
output_path = "/content/ml3d_msn/data/ModelNet10/point_clouds/complete"
if not os.path.exists(output_path):
    os.makedirs(output_path)

for model_category in os.listdir(data_path):
    model_category_path = os.path.join(data_path, model_category)
    for split in ["train", "test"]:
        split_path = os.path.join(model_category_path, split)

        # Check if the split_path is a directory before listing its contents
        if not os.path.isdir(split_path):
            continue

        for model_id in os.listdir(split_path):
            # Skip .ipynb_checkpoints directory
            if model_id.startswith(".ipynb_checkpoints"):
                continue

            if ".off" in model_id:
                model_id_path = os.path.join(split_path, model_id)
                mesh = trimesh.load_mesh(model_id_path)
                mesh.apply_scale(1.0 / max(mesh.extents))
                pcd = PointCloud()
                points, _ = trimesh.sample.sample_surface(mesh, 10000)
                pcd.points = Vector3dVector(points)
                o3d.io.write_point_cloud(os.path.join(output_path,
                                                      model_category + "_" + split + "_" + model_id.split(".")[0] + ".pcd"), pcd)
