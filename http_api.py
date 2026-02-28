import glob
import json
import os
import shutil
import subprocess
import threading
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

app = FastAPI(title="FastGS HTTP API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state: dict[str, Any] = {
    "training": False,
    "last_error": None,
    "model_path": None,
    "source_path": None,
    "job_id": None,
    "started_at": None,
    "log_path": None,
}

_sessions: dict[str, dict[str, Any]] = {}
_run_queue = SingleWorkerTaskQueue()


class RenderPoseSessionRequest(BaseModel):
    session_id: str = ""
    pose_path: str = ""


class RunRequest(BaseModel):
    source_path: str = Field(default="", description="Dataset path inside container")
    model_path: str = Field(default="", description="Output model path or existing model path")
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
    pose_path: str = ""
    frames: int = 240
    ease: bool = True
    render_only: bool = False
    session_id: str = ""
    request_id: str = ""


class DownloadRequest(BaseModel):
    session_id: str = ""
    include_ply: bool = True
    include_video: bool = True
    include_pose: bool = True


def _workspace_root() -> str:
    return os.getenv("FASTGS_WORKSPACE", "/workspace")


def _session_dir(session_id: str) -> str:
    base_dir = os.path.join(_workspace_root(), "tmp_fastgs")
    os.makedirs(base_dir, exist_ok=True)
    session_path = os.path.join(base_dir, session_id)
    os.makedirs(session_path, exist_ok=True)
    return session_path


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
        "dataset_path": "",
        "uploaded_model_path": "",
        "model_path": "",
        "log_path": "",
        "package_status": "idle",
        "package_path": "",
        "package_error": "",
        "pose_renders": [],
        "last_request_id": "",
    }
    _sessions[sid] = session
    return session


def _create_session() -> dict[str, Any]:
    sid = str(uuid.uuid4())
    return _get_session(sid, create_if_missing=True)


def _safe_path(target_path: str, base_dir: str | None = None) -> str:
    root = os.path.abspath(base_dir or _workspace_root())
    target = os.path.abspath(target_path)
    if not (target == root or target.startswith(root + os.sep)):
        raise HTTPException(status_code=400, detail=f"Path must be under {root}")
    return target


def _safe_extract_zip(zip_path: str, extract_dir: str) -> str:
    os.makedirs(extract_dir, exist_ok=True)
    root = os.path.abspath(extract_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            member_path = os.path.abspath(os.path.join(extract_dir, info.filename))
            if not (member_path == root or member_path.startswith(root + os.sep)):
                raise HTTPException(status_code=400, detail="Unsafe zip path")
        zf.extractall(extract_dir)
    return extract_dir


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


def _package_selected_outputs(session_id: str, include_ply: bool = True, include_video: bool = True, include_pose: bool = True) -> str:
    session = _get_session(session_id, create_if_missing=False)
    if not session:
        raise ValueError("Invalid session_id")

    model_path = session.get("model_path", "")
    if not model_path or not os.path.exists(model_path):
        raise ValueError("Model output not found")

    files_to_add: list[tuple[str, str]] = []

    if include_ply:
        ply_path = _latest_point_cloud(model_path)
        if ply_path:
            rel = os.path.join("point_cloud", os.path.basename(os.path.dirname(ply_path)), "point_cloud.ply")
            files_to_add.append((ply_path, rel))

    if include_video:
        video_path = os.path.join(model_path, "orbit_360", "orbit_360.mp4")
        if os.path.exists(video_path):
            files_to_add.append((video_path, "orbit_360.mp4"))

    if include_pose:
        pose_candidates = [p for p in session.get("pose_renders", []) if os.path.exists(p)]
        if pose_candidates:
            pose_path = pose_candidates[-1]
            files_to_add.append((pose_path, os.path.basename(pose_path)))

    if not files_to_add:
        raise ValueError("No output files matched selected options")

    package_path = os.path.join(model_path, "output_package.zip")
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in files_to_add:
            zf.write(src, arc)

    session["package_path"] = package_path
    return package_path


def _package_thread(session_id: str, include_ply: bool = True, include_video: bool = True, include_pose: bool = True) -> None:
    session = _get_session(session_id, create_if_missing=False)
    if not session:
        return

    session["package_status"] = "packaging"
    session["package_error"] = ""
    try:
        _package_selected_outputs(
            session_id,
            include_ply=include_ply,
            include_video=include_video,
            include_pose=include_pose,
        )
        session["package_status"] = "ready"
    except Exception as exc:  # pragma: no cover
        session["package_error"] = str(exc)
        session["package_status"] = "error"


def _run_request_to_dict(req: RunRequest) -> dict[str, Any]:
    if hasattr(req, "model_dump"):
        return req.model_dump()  # pydantic v2
    return req.dict()  # pydantic v1


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

    try:
        source_path = (req.source_path or "").strip() or (session.get("dataset_path", "") or "")
        if not source_path:
            raise RuntimeError("Dataset path missing. Please upload dataset first.")
        if not os.path.exists(source_path):
            raise RuntimeError(f"Dataset path not found: {source_path}")

        if req.render_only:
            model_path = (req.model_path or "").strip() or (session.get("uploaded_model_path", "") or "") or (session.get("model_path", "") or "")
            if not model_path:
                raise RuntimeError("render_only=true requires an existing model path")
            if not os.path.exists(model_path):
                raise RuntimeError(f"Model path not found: {model_path}")
        else:
            model_path = (req.model_path or "").strip() or os.path.join(session["session_dir"], "output")
            os.makedirs(model_path, exist_ok=True)

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
        session["log_path"] = log_path
        session["model_path"] = model_path
        session["last_request_id"] = request_id
        session["package_status"] = "idle"
        session["package_path"] = ""
        session["package_error"] = ""

        _state["log_path"] = log_path
        _state["model_path"] = model_path
        _state["source_path"] = source_path

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

                dataset = lp.extract(args)
                scene, gaussians = _load_scene(dataset, args)
                pipeline = pp.extract(args)

                if req.pose_path:
                    if not os.path.exists(req.pose_path):
                        raise RuntimeError(f"Pose path not found: {req.pose_path}")
                    pose_output = os.path.join(model_path, "renders", f"pose_{int(time.time())}.png")
                    _render_pose_to_output(scene, gaussians, pipeline, dataset, args, req.pose_path, pose_output)
                    session["pose_renders"].append(pose_output)

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

        _state["last_error"] = None
        return {
            "session_id": session["session_id"],
            "model_path": model_path,
            "log_path": log_path,
            "output_ready": True,
        }
    except Exception as exc:
        _state["last_error"] = str(exc)
        raise
    finally:
        _state["training"] = False


_run_queue.start(_execute_run_request)


@app.get("/")
def index():
    index_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(index_path)


@app.get("/health")
def health():
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
    _state["session_id"] = session["session_id"]
    return {
        "session_id": session["session_id"],
        "session_dir": session["session_dir"],
    }


@app.post("/cleanup")
def cleanup(session_id: str = Form(...)):
    session = _get_session(session_id, create_if_missing=False)
    session_dir = session["session_dir"] if session else _session_dir(session_id)
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir, ignore_errors=True)
    _sessions.pop(session_id, None)
    return {"status": "cleaned"}


@app.get("/logs")
def logs(max_lines: int = 200, session_id: str = ""):
    log_path = ""
    if session_id:
        session = _get_session(session_id, create_if_missing=False)
        if session:
            log_path = session.get("log_path", "")
    else:
        log_path = _state.get("log_path") or ""

    if not log_path or not os.path.exists(log_path):
        return {"log": ""}

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()[-max_lines:]
    return {"log": "".join(lines)}


@app.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...), dataset_name: str = Form("dataset"), session_id: str = Form("")):
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip dataset supported")

    session = _get_session(session_id, create_if_missing=True) if session_id else _create_session()
    base_dir = session["session_dir"]

    zip_path = os.path.join(base_dir, f"{dataset_name}.zip")
    with open(zip_path, "wb") as out:
        out.write(await file.read())

    extract_dir = os.path.join(base_dir, dataset_name)
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir, ignore_errors=True)
    _safe_extract_zip(zip_path, extract_dir)

    session["dataset_path"] = extract_dir
    return {
        "session_id": session["session_id"],
        "dataset_path": extract_dir,
    }


@app.post("/upload_model")
async def upload_model(file: UploadFile = File(...), model_name: str = Form("model"), session_id: str = Form("")):
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip model supported")

    session = _get_session(session_id, create_if_missing=True) if session_id else _create_session()
    base_dir = session["session_dir"]

    zip_path = os.path.join(base_dir, f"{model_name}.zip")
    with open(zip_path, "wb") as out:
        out.write(await file.read())

    extract_dir = os.path.join(base_dir, model_name)
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir, ignore_errors=True)
    _safe_extract_zip(zip_path, extract_dir)

    session["uploaded_model_path"] = extract_dir
    session["model_path"] = extract_dir
    return {
        "session_id": session["session_id"],
        "model_path": extract_dir,
    }


@app.post("/upload_pose")
async def upload_pose(file: UploadFile = File(...), session_id: str = Form("")):
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json pose supported")

    session = _get_session(session_id, create_if_missing=True) if session_id else _create_session()
    base_dir = session["session_dir"]

    pose_name = f"pose_{int(time.time())}.json"
    pose_path = os.path.join(base_dir, pose_name)
    with open(pose_path, "wb") as out:
        out.write(await file.read())

    return {
        "session_id": session["session_id"],
        "pose_path": pose_path,
    }


@app.post("/render_pose_session")
def render_pose_session(req: RenderPoseSessionRequest):
    session = _get_session(req.session_id, create_if_missing=False)
    if not session:
        raise HTTPException(status_code=400, detail="Invalid session")

    model_path = session.get("model_path", "") or session.get("uploaded_model_path", "")
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model path not found")

    pose_path = (req.pose_path or "").strip()
    if not pose_path:
        raise HTTPException(status_code=400, detail="pose_path missing")
    if not os.path.exists(pose_path):
        raise HTTPException(status_code=404, detail="pose_path not found")

    args, lp, _, pp = _build_args()
    args.source_path = session.get("dataset_path", "")
    args.model_path = model_path

    dataset = lp.extract(args)
    scene, gaussians = _load_scene(dataset, args)
    pipeline = pp.extract(args)

    output_path = os.path.join(model_path, "renders", f"pose_{int(time.time())}.png")
    _render_pose_to_output(scene, gaussians, pipeline, dataset, args, pose_path, output_path)
    session["pose_renders"].append(output_path)
    return {"pose_path": output_path}


@app.post("/run")
def run(req: RunRequest):
    sid = (req.session_id or "").strip()
    session = _get_session(sid, create_if_missing=True) if sid else _create_session()

    req_data = _run_request_to_dict(req)
    req_data["session_id"] = session["session_id"]

    source_path = (req_data.get("source_path") or "").strip() or (session.get("dataset_path") or "").strip()
    if not source_path:
        raise HTTPException(status_code=409, detail="Dataset not uploaded")

    render_only = bool(req_data.get("render_only"))
    if render_only:
        has_model = (
            (req_data.get("model_path") or "").strip()
            or (session.get("uploaded_model_path") or "").strip()
            or (session.get("model_path") or "").strip()
        )
        if not has_model:
            raise HTTPException(status_code=409, detail="render_only requires uploaded or existing model")

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


@app.post("/prepare_download")
def prepare_download(req: DownloadRequest):
    session = _get_session(req.session_id, create_if_missing=False)
    if not session:
        raise HTTPException(status_code=400, detail="Invalid session")

    if session.get("package_status") == "packaging":
        return {"status": "packaging"}

    threading.Thread(
        target=_package_thread,
        args=(req.session_id, req.include_ply, req.include_video, req.include_pose),
        daemon=True,
    ).start()
    return {"status": "packaging"}


@app.get("/package_status")
def package_status(session_id: str):
    session = _get_session(session_id, create_if_missing=False)
    if not session:
        raise HTTPException(status_code=400, detail="Invalid session")

    return {
        "status": session.get("package_status", "idle"),
        "error": session.get("package_error", ""),
    }


@app.get("/download_selected")
def download_selected(session_id: str):
    session = _get_session(session_id, create_if_missing=False)
    if not session:
        raise HTTPException(status_code=400, detail="Invalid session")

    if session.get("package_status") != "ready":
        raise HTTPException(status_code=409, detail="Package not ready")

    package_path = session.get("package_path", "")
    if not package_path or not os.path.exists(package_path):
        raise HTTPException(status_code=404, detail="Package missing")

    return FileResponse(package_path, filename=os.path.basename(package_path))


@app.get("/download_output")
def download_output(model_path: str = "", session_id: str = ""):
    output_dir = model_path
    if not output_dir and session_id:
        session = _get_session(session_id, create_if_missing=False)
        if session:
            output_dir = session.get("model_path", "")
    if not output_dir:
        output_dir = _state.get("model_path") or ""

    if not output_dir:
        raise HTTPException(status_code=400, detail="No output path available")

    output_dir = _safe_path(output_dir)
    if not os.path.exists(output_dir):
        raise HTTPException(status_code=404, detail="Output path not found")

    zip_path = os.path.join(output_dir, "output_package.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(output_dir):
            for name in files:
                fp = os.path.join(root, name)
                rel = os.path.relpath(fp, output_dir)
                zf.write(fp, rel)

    return FileResponse(zip_path, filename="output_package.zip")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("FASTGS_HOST", "0.0.0.0")
    port = int(os.getenv("FASTGS_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
