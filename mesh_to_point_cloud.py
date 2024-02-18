import pyrender
import trimesh
import numpy as np
from trimesh.transformations import rotation_matrix


T_OPENGL_TO_OPENCV = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
)
VIEWPORT_WIDTH = 600
VIEWPORT_HEIGHT = 600
import shutil


def opengl_projection_matrix_to_intrinsics(P: np.ndarray, width: int, height: int):
    """Convert OpenGL projection matrix to camera intrinsics.
    Args:
        P (np.ndarray): OpenGL projection matrix.
        width (int): Image width.
        height (int): Image height.
    Returns:
        np.ndarray: Camera intrinsics. [3, 3]
    """

    fx = P[0, 0] * width / 2
    fy = P[1, 1] * height / 2
    cx = (1.0 - P[0, 2]) * width / 2
    cy = (1.0 + P[1, 2]) * height / 2

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return K


def depth_to_pointcloud(K: np.ndarray, depth: np.ndarray, rgb: np.ndarray = None):
    """Convert depth image to pointcloud given camera intrinsics.
    Args:
        depth (np.ndarray): Depth image.
    Returns:
        np.ndarray: (x, y, z) Point cloud. [n, 4]
        np.ndarray: (r, g, b) RGB colors per point. [n, 3] or None
    """
    _fx = K[0, 0]
    _fy = K[1, 1]
    _cx = K[0, 2]
    _cy = K[1, 2]

    # Mask out invalid depth
    mask = np.where(depth > 0)
    x, y = mask[1], mask[0]

    # Normalize pixel coordinates
    normalized_x = x.astype(np.float32) - _cx
    normalized_y = y.astype(np.float32) - _cy

    # Convert to world coordinates
    world_x = normalized_x * depth[y, x] / _fx
    world_y = normalized_y * depth[y, x] / _fy
    world_z = depth[y, x]

    pc = np.vstack((world_x, world_y, world_z)).T

    # Assign rgb colors to points if available
    if rgb is not None:
        rgb = rgb[y, x, :]

    return pc, rgb


def render_offscreen(
    mesh: pyrender.Mesh, camera: pyrender.Camera, camera_pose: np.ndarray
):
    """Render mesh offscreen.
    Args:
        mesh (pyrender.Mesh): Mesh object.
        camera (pyrender.Camera): Camera object.
        camera_pose (np.ndarray): Camera pose.
    Returns:
        np.ndarray:  Rendered color image. [H, W, 3]
        np.ndarray:  Rendered depth image. [H, W]
    """
    # Scene
    scene = pyrender.Scene()

    # Mesh
    scene.add(mesh)

    # Camera- pose is w.r.t the scene frame (alternatively, set the parent node)
    scene.add(camera, pose=camera_pose, parent_node=None)

    # Light
    light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=0.5,
        innerConeAngle=np.pi / 16.0,
        outerConeAngle=np.pi / 6.0,
    )
    scene.add(light, pose=camera_pose)

    # Render
    r = pyrender.OffscreenRenderer(VIEWPORT_WIDTH, VIEWPORT_HEIGHT)
    color, depth = r.render(scene)
    return color, depth
import tempfile
def mesh_to_point_cloud(in_file,out_file_partial,out_file_target):
    f = f"{tempfile.gettempdir()}/temp.obj"
    open(f, "w").writelines([l.strip()+"\n" for l in open(in_file).readlines() if l[0] == "f" or l[0] == "v"])
    # Mesh OBJ
    mesh = trimesh.load_mesh(f)
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
    created = 0
    tried = 0
    while created < 50 and tried < 200:
        try:
            # Camera model
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 2, aspectRatio=1.0)

            # Random Camera Pose - This pose is w.r.t the scene parent node
            camera_pose_scene = np.eye(4)
            camera_pose_scene[:3, 3] = [0.1, 0.1, 0.5]

            # Find camera pose transform that rotates the object randomly around its frame
            random_rotation = rotation_matrix(
                (np.random.random() - 0.5) * (2 * np.pi), np.random.random(3)
            )
            obj_pose_camera = np.linalg.inv(camera_pose_scene) @ random_rotation
            camera_pose_scene = np.linalg.inv(obj_pose_camera)

            # Render
            color, depth = render_offscreen(mesh_pyrender, camera, camera_pose_scene)

            # Get Intrinsic Matrix
            K = opengl_projection_matrix_to_intrinsics(
                camera.get_projection_matrix(), width=VIEWPORT_WIDTH, height=VIEWPORT_HEIGHT
            )

            # Deproject depth to point cloud
            point_cloud, _ = depth_to_pointcloud(K, depth)

            # Convert from opengl to opencv camera frame.
            # (OpenGL camera has principal axis along -z, OpenCV has principal axis along +z)
            cam_pose_z_opencv = camera_pose_scene @ T_OPENGL_TO_OPENCV

            # Transform partial point cloud to world coordinates
            point_cloud_h = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
            point_cloud_world = (cam_pose_z_opencv @ point_cloud_h.T).T
            point_cloud_world = point_cloud_world[:, :3]

            # Visualize alignment of partial point cloud in the world frame
            resampleinds = np.random.choice(point_cloud_world.shape[0],8092)
            point_cloud_world = point_cloud_world[resampleinds]
            complete = trimesh.sample.sample_surface(mesh, 8092)[0]
            trimesh.points.PointCloud(point_cloud_world).export(f"{out_file_partial}_{created}_denoised.obj")
            trimesh.points.PointCloud(complete).export(f"{out_file_target}_{created}_denoised.obj")
            created += 1
            tried += 1
        except:
            tried += 1
            print(tried)
            continue