import os
import numpy as np
import trimesh
import open3d as o3d

def random_pose():
    # Function to generate a random pose matrix (4x4) for the model
    angle_x = np.random.uniform() * 2 * np.pi
    angle_y = np.random.uniform() * 2 * np.pi
    angle_z = np.random.uniform() * 2 * np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    t = np.random.rand(3) * 2 - 1  # Random translation in the range [-1, 1]
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose

def generate_point_cloud(mesh, num_points):
    # Function to generate a point cloud from a given mesh
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points

if __name__ == '__main__':
    model_dir = "/content/ml3d_msn/data/ModelNet10/ModelNet10"  # Replace with the path to ModelNet10 dataset
    output_dir = "/content/ml3d_msn/data/ModelNet10/point_clouds"  # Replace with the desired output directory
    num_scans = 50
    num_points = 10000

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    category_list = os.listdir(model_dir)
    print("Listing categories:")
    category_list = [category for category in category_list if category != ".ipynb_checkpoints" and category != ".DS_Store" and category != "README.txt" ]
    [print(x) for x in category_list]

    for category in category_list:
        for split in ['train', 'test']:
            model_list = os.listdir(os.path.join(model_dir, category, split))
            for model in model_list:
                model_id = model.split(".")[0]
                if len(model_id) == 0 or not model.endswith(".off") or model.startswith(".DS_Store"):
                    continue
                
                # Import mesh model using trimesh
                model_path = os.path.join(model_dir, category, split, model)
                mesh = trimesh.load_mesh(model_path)
                mesh.apply_scale(1.0 / max(mesh.extents))

                # Generate synthetic point clouds
                for i in range(num_scans):
                    pose = random_pose()
                    transformed_mesh = mesh.copy()
                    transformed_mesh.apply_transform(pose)

                    # Generate point cloud using trimesh
                    points = generate_point_cloud(transformed_mesh, num_points)

                    # Convert to Open3D point cloud
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)

                    # Save point cloud to disk
                    output_filename = f"{category}_{split}_{model_id}_{i}.pcd"
                    output_path = os.path.join(output_dir, output_filename)
                    o3d.io.write_point_cloud(output_path, pcd)

                print(f'{model_id} done')

    # Save intrinsics.txt file
    width = 160  # Replace with the appropriate width value
    height = 120  # Replace with the appropriate height value
    focal = 100  # Replace with the appropriate focal length value
    intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    np.savetxt(os.path.join(output_dir, 'intrinsics.txt'), intrinsics, '%f')

    print("Point cloud generation complete.")
