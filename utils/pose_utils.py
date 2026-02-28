import json
import numpy as np
import torch

from scene.cameras import Camera


def _normalize(vec):
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        raise ValueError("Pose vector has near-zero length.")
    return vec / norm


def load_pose_json(pose_json_path):
    with open(pose_json_path, "r") as pose_file:
        data = json.load(pose_file)

    if "R" in data and "T" in data:
        r_mat = np.array(data["R"], dtype=np.float32)
        t_vec = np.array(data["T"], dtype=np.float32)
        return r_mat, t_vec

    if "w2c" in data:
        w2c = np.array(data["w2c"], dtype=np.float32)
        r_mat = w2c[:3, :3]
        t_vec = w2c[:3, 3]
        return r_mat, t_vec

    if "c2w" in data:
        c2w = np.array(data["c2w"], dtype=np.float32)
        w2c = np.linalg.inv(c2w)
        r_mat = w2c[:3, :3]
        t_vec = w2c[:3, 3]
        return r_mat, t_vec

    raise ValueError("Pose JSON must contain R/T, w2c, or c2w.")


def build_camera_from_pose(ref_cam, r_mat, t_vec, data_device="cuda", uid=0, image_name="pose"):
    height = int(ref_cam.image_height)
    width = int(ref_cam.image_width)
    dummy_image = torch.zeros((3, height, width), dtype=torch.float32)
    return Camera(
        colmap_id=uid,
        R=r_mat,
        T=t_vec,
        FoVx=ref_cam.FoVx,
        FoVy=ref_cam.FoVy,
        image=dummy_image,
        gt_alpha_mask=None,
        image_name=image_name,
        uid=uid,
        data_device=data_device,
    )


def build_look_at_pose(camera_pos, target, up=np.array([0.0, 1.0, 0.0], dtype=np.float32)):
    forward = _normalize(target - camera_pos)
    z_axis = _normalize(-forward)
    x_axis = _normalize(np.cross(up, z_axis))
    y_axis = np.cross(z_axis, x_axis)

    r_mat = np.stack([x_axis, y_axis, z_axis], axis=0).astype(np.float32)
    t_vec = (-r_mat @ camera_pos.astype(np.float32)).astype(np.float32)
    return r_mat, t_vec


def build_camera_from_pose_with_fov(ref_cam, r_mat, t_vec, fovx, fovy, data_device="cuda", uid=0, image_name="pose"):
    height = int(ref_cam.image_height)
    width = int(ref_cam.image_width)
    dummy_image = torch.zeros((3, height, width), dtype=torch.float32)
    return Camera(
        colmap_id=uid,
        R=r_mat,
        T=t_vec,
        FoVx=float(fovx),
        FoVy=float(fovy),
        image=dummy_image,
        gt_alpha_mask=None,
        image_name=image_name,
        uid=uid,
        data_device=data_device,
    )


def _rotmat_to_quat(r_mat):
    m = r_mat.astype(np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-8
    return q.astype(np.float32)


def _quat_to_rotmat(q):
    q = q.astype(np.float64)
    q /= np.linalg.norm(q) + 1e-8
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def _slerp_quat(q0, q1, t):
    q0 = q0.astype(np.float64)
    q1 = q1.astype(np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        result /= np.linalg.norm(result) + 1e-8
        return result.astype(np.float32)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / (sin_theta_0 + 1e-8)
    s1 = np.sin(theta) / (sin_theta_0 + 1e-8)
    result = (s0 * q0) + (s1 * q1)
    result /= np.linalg.norm(result) + 1e-8
    return result.astype(np.float32)


def interpolate_camera_params(cameras, interp_per_pair=3, loop=True):
    if cameras is None or len(cameras) == 0:
        return []
    if len(cameras) == 1:
        cam = cameras[0]
        return [{
            "R": cam.R,
            "T": cam.T,
            "FovX": cam.FoVx,
            "FovY": cam.FoVy,
            "image_name": cam.image_name,
        }]

    params_list = []
    cam_count = len(cameras)
    pair_count = cam_count if loop else cam_count - 1
    for idx in range(pair_count):
        cam_a = cameras[idx]
        cam_b = cameras[(idx + 1) % cam_count]
        params_list.append({
            "R": cam_a.R,
            "T": cam_a.T,
            "FovX": cam_a.FoVx,
            "FovY": cam_a.FoVy,
            "image_name": cam_a.image_name,
        })

        qa = _rotmat_to_quat(np.array(cam_a.R))
        qb = _rotmat_to_quat(np.array(cam_b.R))
        ta = np.array(cam_a.T, dtype=np.float32)
        tb = np.array(cam_b.T, dtype=np.float32)
        fovx_a = float(cam_a.FoVx)
        fovx_b = float(cam_b.FoVx)
        fovy_a = float(cam_a.FoVy)
        fovy_b = float(cam_b.FoVy)

        for step in range(1, interp_per_pair + 1):
            t = step / float(interp_per_pair + 1)
            q = _slerp_quat(qa, qb, t)
            r_mat = _quat_to_rotmat(q)
            t_vec = (1.0 - t) * ta + t * tb
            fovx = (1.0 - t) * fovx_a + t * fovx_b
            fovy = (1.0 - t) * fovy_a + t * fovy_b
            params_list.append({
                "R": r_mat,
                "T": t_vec,
                "FovX": fovx,
                "FovY": fovy,
                "image_name": f"{cam_a.image_name}_interp_{step}",
            })

    if not loop:
        cam_last = cameras[-1]
        params_list.append({
            "R": cam_last.R,
            "T": cam_last.T,
            "FovX": cam_last.FoVx,
            "FovY": cam_last.FoVy,
            "image_name": cam_last.image_name,
        })
    return params_list
