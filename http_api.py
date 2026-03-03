"""
FastGS GPU Backend API

This backend handles training/render + OSS I/O only.
Task recover/pause is managed by suanli-task-manager.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import time
import uuid
import zipfile
from argparse import ArgumentParser
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

import numpy as np
import torch
import torchvision
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import GaussianModel, render_fastgs
from scene import Scene
from task_queue import SingleWorkerTaskQueue
from train import training
from utils.general_utils import safe_state
from utils.pose_utils import (
    build_camera_from_pose,
    build_camera_from_pose_with_fov,
    build_look_at_pose,
    interpolate_camera_params,
)

app = FastAPI(title="FastGS HTTP API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {
    "training": False,
    "last_error": None,
    "job_id": "",
    "started_at": None,
    "log_path": "",
}

_sessions: dict[str, dict[str, Any]] = {}
_run_queue = SingleWorkerTaskQueue()

# ---------------------------------------------------------------------------
# OSS config
# ---------------------------------------------------------------------------
OSS_BUCKET = os.getenv("FASTGS_OSS_BUCKET", "kokokoni")
OSS_PREFIX = os.getenv("FASTGS_OSS_PREFIX", "docker-input&output/fastgs").strip("/")
OSS_SIGN_EXPIRES = os.getenv("FASTGS_OSS_SIGN_EXPIRES", "24h")


class RunRequest(BaseModel):
    source_path: str = Field(default="", description="Dataset OSS key override")
    model_path: str = Field(default="", description="Model OSS key override")
    iterations: int = 30_000
    mult: float = 0.5
    white_background: bool = False
    eval: bool = False
    data_device: str = "cuda"
    iteration: int = -1
    video360: bool = False
    use_dataset_cams: bool = False
    use_test_cams: bool = False
    interp_per_pair: int = 3
    loop: bool = False
    fps: int = 30
    pose_path: str = ""  # pose OSS key
    frames: int = 240
    ease: bool = True
    render_only: bool = False
    session_id: str = ""
    request_id: str = ""


# ---------------------------------------------------------------------------
# Path/session helpers
# ---------------------------------------------------------------------------
def _tmp_root() -> str:
    root = os.getenv("FASTGS_TMP_ROOT", "/tmp/fastgs-oss-workspace")
    os.makedirs(root, exist_ok=True)
    return root


def _sessions_root() -> str:
    root = os.path.join(_tmp_root(), "sessions")
    os.makedirs(root, exist_ok=True)
    return root


def _runs_root() -> str:
    root = os.path.join(_tmp_root(), "runs")
    os.makedirs(root, exist_ok=True)
    return root


def _session_dir(session_id: str) -> str:
    path = os.path.join(_sessions_root(), session_id)
    os.makedirs(path, exist_ok=True)
    return path


def _run_dir(session_id: str, request_id: str) -> str:
    path = os.path.join(_runs_root(), session_id, request_id)
    os.makedirs(path, exist_ok=True)
    return path


def _get_session(session_id: str, create_if_missing: bool = False) -> dict[str, Any] | None:
    sid = (session_id or "").strip()
    if not sid:
        return None
    if sid in _sessions:
        return _sessions[sid]
    if not create_if_missing:
        return None

    session = {
        "session_id": sid,
        "session_dir": _session_dir(sid),
        "dataset_oss_key": "",
        "model_oss_key": "",
        "pose_oss_keys": [],
        "active_log_path": "",
        "last_request_id": "",
        "last_result": None,
    }
    _sessions[sid] = session
    return session


def _create_session() -> dict[str, Any]:
    sid = str(uuid.uuid4())
    return _get_session(sid, create_if_missing=True)


def _safe_extract_zip(zip_path: str, extract_dir: str) -> str:
    os.makedirs(extract_dir, exist_ok=True)
    root = os.path.abspath(extract_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            member_path = os.path.abspath(os.path.join(extract_dir, info.filename))
            if not (member_path == root or member_path.startswith(root + os.sep)):
                raise RuntimeError("Unsafe zip path")
        zf.extractall(extract_dir)
    return extract_dir


def _resolve_extracted_root(extract_dir: str) -> str:
    entries = [
        os.path.join(extract_dir, name)
        for name in os.listdir(extract_dir)
        if name not in {"__MACOSX", ".DS_Store"}
    ]
    dirs = [path for path in entries if os.path.isdir(path)]
    files = [path for path in entries if os.path.isfile(path)]
    if len(dirs) == 1 and not files:
        return dirs[0]
    return extract_dir


# ---------------------------------------------------------------------------
# OSS helpers
# ---------------------------------------------------------------------------
def _normalize_oss_key(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""

    if raw.startswith("http://") or raw.startswith("https://"):
        raise ValueError("Expected OSS key, not presigned URL")

    if raw.startswith("oss://"):
        payload = raw[len("oss://") :]
        if "/" not in payload:
            return ""
        bucket, key = payload.split("/", 1)
        if bucket and bucket != OSS_BUCKET:
            raise ValueError(f"OSS bucket mismatch: {bucket}")
        return key.lstrip("/")

    return raw.lstrip("/")


def _oss_url(oss_key: str) -> str:
    key = _normalize_oss_key(oss_key)
    if not key:
        raise RuntimeError("Empty OSS key")
    return f"oss://{OSS_BUCKET}/{key}"


def _run_ossutil(args: list[str]) -> str:
    timeout_sec = int(os.getenv("FASTGS_OSSUTIL_TIMEOUT_SEC", "1800"))
    cmd = ["ossutil", *args]
    print(f"[oss] exec: {' '.join(cmd)} (timeout={timeout_sec}s)")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ossutil timeout after {timeout_sec}s: {' '.join(cmd[:3])}") from exc
    except FileNotFoundError as exc:
        raise RuntimeError("ossutil not found in PATH") from exc
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or "unknown ossutil error"
        raise RuntimeError(f"ossutil {' '.join(args[:2])} failed: {detail}")
    return (result.stdout or "").strip()


def _join_oss_key(*parts: str) -> str:
    chunks = []
    for part in parts:
        p = str(part or "").strip("/")
        if p:
            chunks.append(p)
    return "/".join(chunks)


def oss_upload_file(local_path: str, oss_key: str) -> None:
    _run_ossutil(["cp", local_path, _oss_url(oss_key), "-f"])


def oss_download_file(oss_key: str, local_path: str) -> None:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    _run_ossutil(["cp", _oss_url(oss_key), local_path, "-f"])


def oss_sign_url(oss_key: str, expires: str = OSS_SIGN_EXPIRES) -> str:
    output = _run_ossutil(["presign", _oss_url(oss_key), "--expires-duration", expires])
    for line in output.splitlines():
        candidate = line.strip()
        if candidate.startswith("https://") or candidate.startswith("http://"):
            return candidate
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    raise RuntimeError("ossutil presign returned empty output")


async def _save_upload_file(file: UploadFile, local_path: str) -> int:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    total = 0
    with open(local_path, "wb") as out:
        while True:
            chunk = await file.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
    await file.close()
    return total


def _download_and_extract_zip(oss_key: str, target_dir: str, zip_name: str) -> str:
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, zip_name)
    extract_dir = os.path.join(target_dir, "extract")

    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir, ignore_errors=True)

    oss_download_file(oss_key, zip_path)
    _safe_extract_zip(zip_path, extract_dir)
    return _resolve_extracted_root(extract_dir)


# ---------------------------------------------------------------------------
# FastGS core helpers
# ---------------------------------------------------------------------------
def _build_args():
    parser = ArgumentParser(description="FastGS API args")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--websockets", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--iteration", type=int, default=-1)

    args = parser.parse_args([])
    return args, lp, op, pp


def _log_path(model_path: str) -> str:
    log_dir = os.path.join(model_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "train.log")


def _load_pose_dict(pose_path: str) -> dict[str, Any]:
    with open(pose_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_pose(pose_dict: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "R" in pose_dict and "T" in pose_dict:
        return np.array(pose_dict["R"], dtype=np.float32), np.array(pose_dict["T"], dtype=np.float32)

    if "w2c" in pose_dict:
        w2c = np.array(pose_dict["w2c"], dtype=np.float32)
        return w2c[:3, :3], w2c[:3, 3]

    if "c2w" in pose_dict:
        c2w = np.array(pose_dict["c2w"], dtype=np.float32)
        w2c = np.linalg.inv(c2w)
        return w2c[:3, :3], w2c[:3, 3]

    raise ValueError("Pose JSON must contain R/T, w2c, or c2w")


def _load_scene(dataset, args):
    gaussians = GaussianModel(dataset.sh_degree, optimizer_type="default")
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    return scene, gaussians


def _render_pose_to_output(scene, gaussians, pipeline, dataset, args, pose_path: str, output_path: str) -> str:
    pose_dict = _load_pose_dict(pose_path)
    r_mat, t_vec = _parse_pose(pose_dict)

    ref_cam = scene.getTrainCameras()[0]
    pose_cam = build_camera_from_pose(ref_cam, r_mat, t_vec, data_device=dataset.data_device)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    rendering = render_fastgs(pose_cam, gaussians, pipeline, background, args.mult)["render"]
    rendering = torch.clamp(rendering, 0.0, 1.0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchvision.utils.save_image(rendering, output_path)
    return output_path


def _render_dataset_video(scene, gaussians, pipeline, dataset, args, output_dir: str, interp_per_pair: int = 3, use_test: bool = False, loop: bool = False) -> str:
    cams = scene.getTestCameras() if use_test else scene.getTrainCameras()
    if not cams:
        cams = scene.getTrainCameras()

    ref_cam = cams[0]
    params_list = interpolate_camera_params(cams, interp_per_pair=interp_per_pair, loop=loop)

    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for idx, params in enumerate(params_list):
        pose_cam = build_camera_from_pose_with_fov(
            ref_cam,
            np.array(params["R"], dtype=np.float32),
            np.array(params["T"], dtype=np.float32),
            params["FovX"],
            params["FovY"],
            data_device=dataset.data_device,
            uid=idx,
            image_name=params.get("image_name", f"interp_{idx}"),
        )
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        rendering = render_fastgs(pose_cam, gaussians, pipeline, background, args.mult)["render"]
        rendering = torch.clamp(rendering, 0.0, 1.0)
        torchvision.utils.save_image(rendering, os.path.join(frames_dir, f"{idx:05d}.png"))

    return frames_dir


def _render_orbit_frames(scene, gaussians, pipeline, dataset, args, output_dir: str, frames: int, ease: bool = True) -> str:
    ref_cam = scene.getTrainCameras()[0]
    cam_centers = np.stack([cam.camera_center.detach().cpu().numpy() for cam in scene.getTrainCameras()], axis=0)
    gauss_center = scene.gaussians.get_xyz.mean(dim=0).detach().cpu().numpy()
    target = gauss_center.astype(np.float32)

    cam_dists = np.linalg.norm(cam_centers - target[None, :], axis=1)
    radius = float(np.median(cam_dists))
    if radius <= 1e-6:
        radius = float(scene.cameras_extent)
    height = float(np.median(cam_centers[:, 1] - target[1]))

    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for idx in range(frames):
        t = idx / float(frames)
        if ease:
            t = 0.5 - 0.5 * np.cos(2.0 * np.pi * t)
        theta = 2.0 * np.pi * t

        camera_pos = (
            target
            + np.array(
                [radius * np.cos(theta), height, radius * np.sin(theta)],
                dtype=np.float32,
            )
        ).astype(np.float32)
        r_mat, t_vec = build_look_at_pose(camera_pos, target)

        pose_cam = build_camera_from_pose(
            ref_cam,
            r_mat,
            t_vec,
            data_device=dataset.data_device,
            uid=idx,
        )
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        rendering = render_fastgs(pose_cam, gaussians, pipeline, background, args.mult)["render"]
        rendering = torch.clamp(rendering, 0.0, 1.0)
        torchvision.utils.save_image(rendering, os.path.join(frames_dir, f"{idx:05d}.png"))

    return frames_dir


def _encode_video(output_dir: str, fps: int) -> str:
    video_path = os.path.join(output_dir, "orbit_360.mp4")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        os.path.join(output_dir, "frames", "%05d.png"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        video_path,
    ]
    subprocess.run(ffmpeg_cmd, check=False)
    return video_path


def _latest_point_cloud(model_path: str) -> str:
    matches = glob.glob(os.path.join(model_path, "point_cloud", "iteration_*", "point_cloud.ply"))
    if not matches:
        return ""

    def _iter_num(path: str) -> int:
        try:
            name = os.path.basename(os.path.dirname(path))
            return int(name.split("iteration_")[-1])
        except Exception:
            return -1

    return sorted(matches, key=_iter_num)[-1]


def _build_bundle(bundle_path: str, files: list[tuple[str, str]]) -> None:
    valid = [(src, arc) for src, arc in files if src and os.path.exists(src)]
    if not valid:
        raise RuntimeError("No artifacts for bundle")

    os.makedirs(os.path.dirname(bundle_path), exist_ok=True)
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in valid:
            zf.write(src, arc)


def _upload_artifact(local_path: str, oss_output_prefix: str, file_name: str) -> dict[str, str] | None:
    if not local_path or not os.path.exists(local_path):
        return None

    oss_key = _join_oss_key(oss_output_prefix, file_name)
    oss_upload_file(local_path, oss_key)
    return {
        "oss_key": oss_key,
        "url": oss_sign_url(oss_key),
    }


def _run_request_to_dict(req: RunRequest) -> dict[str, Any]:
    if hasattr(req, "model_dump"):
        return req.model_dump()
    return req.dict()


def _tail_log(log_path: str, max_lines: int) -> str:
    if not log_path or not os.path.exists(log_path):
        return ""
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()[-max_lines:]
    return "".join(lines)


# ---------------------------------------------------------------------------
# Worker execution
# ---------------------------------------------------------------------------
def _execute_run_request(request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = RunRequest(**payload)

    _state["training"] = True
    _state["last_error"] = None
    _state["job_id"] = request_id
    _state["started_at"] = time.time()

    session = _get_session(req.session_id, create_if_missing=True)
    if not session:
        _state["training"] = False
        raise RuntimeError("Invalid session")

    run_dir = ""
    try:
        run_dir = _run_dir(session["session_id"], request_id)
        input_dir = os.path.join(run_dir, "input")
        os.makedirs(input_dir, exist_ok=True)

        source_key = _normalize_oss_key((req.source_path or "").strip() or session.get("dataset_oss_key", ""))
        model_key = _normalize_oss_key((req.model_path or "").strip() or session.get("model_oss_key", ""))
        pose_key = _normalize_oss_key((req.pose_path or "").strip()) if req.pose_path else ""

        if not req.render_only and not source_key:
            raise RuntimeError("Dataset OSS key missing. Upload dataset first.")
        if req.render_only and not model_key:
            raise RuntimeError("render_only=true requires model OSS key.")

        source_path = ""
        if source_key:
            source_path = _download_and_extract_zip(source_key, os.path.join(input_dir, "dataset"), "dataset.zip")
        elif not req.render_only:
            raise RuntimeError("Dataset OSS key missing. Upload dataset first.")

        if req.render_only:
            model_path = _download_and_extract_zip(model_key, os.path.join(input_dir, "model"), "model.zip")
        else:
            model_path = os.path.join(run_dir, "output")
            os.makedirs(model_path, exist_ok=True)

        local_pose_path = ""
        if pose_key:
            local_pose_path = os.path.join(input_dir, "pose", f"pose_{int(time.time())}.json")
            oss_download_file(pose_key, local_pose_path)

        args, lp, op, pp = _build_args()
        args.source_path = source_path
        args.model_path = model_path
        args.iterations = req.iterations
        args.mult = req.mult
        args.white_background = req.white_background
        args.eval = req.eval
        args.data_device = req.data_device
        args.iteration = req.iteration
        args.save_iterations.append(args.iterations)

        safe_state(args.quiet)

        log_path = _log_path(model_path)
        session["active_log_path"] = log_path
        session["last_request_id"] = request_id
        _state["log_path"] = log_path

        with open(log_path, "a", encoding="utf-8") as log_f:
            with redirect_stdout(log_f), redirect_stderr(log_f):
                if not req.render_only:
                    training(
                        lp.extract(args),
                        op.extract(args),
                        pp.extract(args),
                        args.test_iterations,
                        args.save_iterations,
                        args.checkpoint_iterations,
                        args.start_checkpoint,
                        args.debug_from,
                        args.websockets,
                    )

                pose_output = ""
                needs_scene = (not req.render_only) or bool(local_pose_path) or bool(req.video360)
                if needs_scene:
                    if not source_path:
                        raise RuntimeError("Pose/video rendering requires dataset OSS input.")

                    dataset = lp.extract(args)
                    scene, gaussians = _load_scene(dataset, args)
                    pipeline = pp.extract(args)

                    if local_pose_path:
                        pose_output = os.path.join(model_path, "renders", f"pose_{int(time.time())}.png")
                        _render_pose_to_output(scene, gaussians, pipeline, dataset, args, local_pose_path, pose_output)

                    if req.video360:
                        output_dir = os.path.join(model_path, "orbit_360")
                        if req.use_dataset_cams:
                            _render_dataset_video(
                                scene,
                                gaussians,
                                pipeline,
                                dataset,
                                args,
                                output_dir,
                                interp_per_pair=req.interp_per_pair,
                                use_test=req.use_test_cams,
                                loop=req.loop,
                            )
                        else:
                            _render_orbit_frames(
                                scene,
                                gaussians,
                                pipeline,
                                dataset,
                                args,
                                output_dir,
                                req.frames,
                                req.ease,
                            )
                        _encode_video(output_dir, req.fps)
                else:
                    print("render_only with no dataset: skip pose/video rendering")

        ply_path = _latest_point_cloud(model_path)
        if not ply_path:
            raise RuntimeError("point_cloud.ply not found in output")

        video_path = os.path.join(model_path, "orbit_360", "orbit_360.mp4")
        if not os.path.exists(video_path):
            video_path = ""

        pose_output_path = ""
        pose_candidates = glob.glob(os.path.join(model_path, "renders", "pose_*.png"))
        if pose_candidates:
            pose_output_path = sorted(pose_candidates)[-1]

        bundle_path = os.path.join(run_dir, "fastgs_output.zip")
        _build_bundle(
            bundle_path,
            [
                (ply_path, "point_cloud.ply"),
                (video_path, "orbit_360.mp4"),
                (pose_output_path, os.path.basename(pose_output_path) if pose_output_path else "pose.png"),
                (log_path, "train.log"),
            ],
        )

        oss_prefix_root = _join_oss_key(OSS_PREFIX, session["session_id"], request_id)
        oss_output_prefix = _join_oss_key(oss_prefix_root, "output")

        artifacts = {
            "ply": _upload_artifact(ply_path, oss_output_prefix, "point_cloud.ply"),
            "video": _upload_artifact(video_path, oss_output_prefix, "orbit_360.mp4") if video_path else None,
            "pose": _upload_artifact(
                pose_output_path,
                oss_output_prefix,
                os.path.basename(pose_output_path),
            ) if pose_output_path else None,
            "bundle": _upload_artifact(bundle_path, oss_output_prefix, "fastgs_output.zip"),
            "log": _upload_artifact(log_path, oss_output_prefix, "train.log"),
        }

        result = {
            "session_id": session["session_id"],
            "oss_prefix": oss_prefix_root,
            "artifacts": artifacts,
        }

        session["last_result"] = result
        _state["last_error"] = None
        return result

    except Exception as exc:
        _state["last_error"] = str(exc)
        raise
    finally:
        session["active_log_path"] = ""
        _state["training"] = False
        if run_dir and os.path.exists(run_dir) and os.getenv("FASTGS_KEEP_TMP", "0") != "1":
            shutil.rmtree(run_dir, ignore_errors=True)


_run_queue.start(_execute_run_request)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    index_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(index_path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    return {"status": "ok"}


@app.get("/live")
def live():
    return {"status": "ok"}


@app.get("/status")
def status():
    data = dict(_state)
    data["sessions"] = len(_sessions)
    data["queue"] = _run_queue.get_queue_status()
    return data


@app.get("/queue_status")
def queue_status(request_id: str = ""):
    rid = request_id.strip()
    return _run_queue.get_queue_status(request_id=rid if rid else None)


@app.get("/request_status")
def request_status(request_id: str):
    rid = request_id.strip()
    if not rid:
        raise HTTPException(status_code=400, detail="request_id is required")
    payload = _run_queue.get_request_status(rid)
    if not payload:
        raise HTTPException(status_code=404, detail="request_id not found")
    return payload


@app.post("/create_session")
def create_session():
    session = _create_session()
    return {
        "session_id": session["session_id"],
    }


@app.post("/cleanup")
def cleanup(session_id: str = Form(...)):
    sid = (session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id is required")

    session_root = os.path.join(_sessions_root(), sid)
    run_root = os.path.join(_runs_root(), sid)
    if os.path.exists(session_root):
        shutil.rmtree(session_root, ignore_errors=True)
    if os.path.exists(run_root):
        shutil.rmtree(run_root, ignore_errors=True)
    _sessions.pop(sid, None)

    return {"status": "cleaned"}


@app.get("/logs")
def logs(max_lines: int = 200, session_id: str = ""):
    log_path = ""
    if session_id:
        session = _get_session(session_id, create_if_missing=False)
        if session:
            log_path = session.get("active_log_path", "")
    else:
        log_path = _state.get("log_path") or ""

    return {"log": _tail_log(log_path, max(1, min(2000, max_lines)))}


@app.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...), session_id: str = Form("")):
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip dataset supported")

    session = _get_session(session_id, create_if_missing=True) if session_id else _create_session()
    local_path = os.path.join(session["session_dir"], "uploads", f"dataset_{uuid.uuid4().hex}.zip")

    try:
        print(f"[upload_dataset] start session={session['session_id']} filename={file.filename}")
        total_bytes = await _save_upload_file(file, local_path)
        print(f"[upload_dataset] saved_local bytes={total_bytes} path={local_path}")
        oss_key = _join_oss_key(OSS_PREFIX, session["session_id"], "input", "dataset.zip")
        oss_upload_file(local_path, oss_key)
        print(f"[upload_dataset] uploaded_oss key={oss_key}")
        session["dataset_oss_key"] = oss_key
        return {
            "session_id": session["session_id"],
            "dataset_oss_key": oss_key,
            "bytes": total_bytes,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)


@app.post("/upload_model")
async def upload_model(file: UploadFile = File(...), session_id: str = Form("")):
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip model supported")

    session = _get_session(session_id, create_if_missing=True) if session_id else _create_session()
    local_path = os.path.join(session["session_dir"], "uploads", f"model_{uuid.uuid4().hex}.zip")

    try:
        print(f"[upload_model] start session={session['session_id']} filename={file.filename}")
        total_bytes = await _save_upload_file(file, local_path)
        print(f"[upload_model] saved_local bytes={total_bytes} path={local_path}")
        oss_key = _join_oss_key(OSS_PREFIX, session["session_id"], "input", "model.zip")
        oss_upload_file(local_path, oss_key)
        print(f"[upload_model] uploaded_oss key={oss_key}")
        session["model_oss_key"] = oss_key
        return {
            "session_id": session["session_id"],
            "model_oss_key": oss_key,
            "bytes": total_bytes,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)


@app.post("/upload_pose")
async def upload_pose(file: UploadFile = File(...), session_id: str = Form("")):
    if not (file.filename or "").lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json pose supported")

    session = _get_session(session_id, create_if_missing=True) if session_id else _create_session()
    local_path = os.path.join(session["session_dir"], "uploads", f"pose_{uuid.uuid4().hex}.json")

    try:
        print(f"[upload_pose] start session={session['session_id']} filename={file.filename}")
        total_bytes = await _save_upload_file(file, local_path)
        print(f"[upload_pose] saved_local bytes={total_bytes} path={local_path}")
        pose_name = f"pose_{int(time.time())}_{uuid.uuid4().hex[:8]}.json"
        oss_key = _join_oss_key(OSS_PREFIX, session["session_id"], "input", "pose", pose_name)
        oss_upload_file(local_path, oss_key)
        print(f"[upload_pose] uploaded_oss key={oss_key}")

        pose_keys: list[str] = session.get("pose_oss_keys", [])
        pose_keys.append(oss_key)
        session["pose_oss_keys"] = pose_keys[-20:]

        return {
            "session_id": session["session_id"],
            "pose_oss_key": oss_key,
            "bytes": total_bytes,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)


@app.post("/run")
def run(req: RunRequest):
    sid = (req.session_id or "").strip()
    session = _get_session(sid, create_if_missing=True) if sid else _create_session()

    req_data = _run_request_to_dict(req)
    req_data["session_id"] = session["session_id"]

    try:
        source_key = _normalize_oss_key((req_data.get("source_path") or "").strip() or session.get("dataset_oss_key", ""))
        model_key = _normalize_oss_key((req_data.get("model_path") or "").strip() or session.get("model_oss_key", ""))

        if source_key:
            req_data["source_path"] = source_key
        if model_key:
            req_data["model_path"] = model_key

        pose_value = (req_data.get("pose_path") or "").strip()
        if pose_value:
            req_data["pose_path"] = _normalize_oss_key(pose_value)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    render_only = bool(req_data.get("render_only"))
    if not render_only and not req_data.get("source_path"):
        raise HTTPException(status_code=409, detail="Dataset not uploaded")
    if render_only and not req_data.get("model_path"):
        raise HTTPException(status_code=409, detail="render_only requires uploaded model")

    try:
        request_id, position = _run_queue.enqueue(
            req_data,
            request_id=(req_data.get("request_id") or "").strip() or None,
            metadata={"session_id": session["session_id"]},
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    session["last_request_id"] = request_id

    return {
        "status": "queued",
        "request_id": request_id,
        "position": position,
        "session_id": session["session_id"],
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("FASTGS_HOST", "0.0.0.0")
    port = int(os.getenv("FASTGS_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
