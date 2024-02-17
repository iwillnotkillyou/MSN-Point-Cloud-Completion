import numpy as np
import trimesh
import pyrender

def depth2pcd(depth, intrinsics, pose, num):
    depth = np.flipud(depth)
    y, x = np.where(depth > 0)
    inds = np.random.choice(np.arange(y.shape[0]), num)
    y, x = y[inds], x[inds]
    tm =  np.linalg.inv(pose) @ intrinsics
    points = np.stack([(x/depth.shape[1])*2-1, (y/depth.shape[0])*2-1, np.ones(x.shape[0]), np.ones(x.shape[0])], 0)
    return (tm @ points).T[:, :3]

def random_pose():
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
    t = np.expand_dims(R[:, 2], 1)
    pose = np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)
    return pose
def mesh_to_point_cloud(path, out_path):
    fuze_trimesh = trimesh.load(path)
    mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose = random_pose()
    intrinsics = camera.get_projection_matrix()
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                               innerConeAngle=np.pi/16.0,
                               outerConeAngle=np.pi/6.0)
    scene.add(light, pose=camera_pose)
    r = pyrender.OffscreenRenderer(400, 400)
    color, depth = r.render(scene)
    points = depth2pcd(depth,intrinsics,camera_pose,8092)
    ls = [f"v {p[0]} {p[1]} {p[2]}\n" for p in points]
    open(out_path,"w").writelines(ls)