"""Compare model-predicted future frames with a LIBERO action rollout."""

import dataclasses
import json
import logging
import math
import pathlib

import imageio.v2 as imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from PIL import Image
from PIL import ImageDraw
import torch
import tyro

from openpi import transforms
from openpi.models import model as _model
from openpi.shared import image_tools
from openpi.training import checkpoints
from openpi.training import config as training_config

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    checkpoint_dir: pathlib.Path = pathlib.Path(
        "checkpoints/pi05_expert_visuotactile_spatiotemporal_libero/libero_spatiotemporal_p2_b8_30k/30000"
    )
    config_name: str = "pi05_expert_visuotactile_spatiotemporal_libero"
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    num_episodes: int = 3
    num_steps_wait: int = 10
    resize_size: int = 224
    sample_steps: int = 10
    rollout_steps: int | None = None  # None executes the full aligned action/future-frame horizon.
    video_fps: int = 10
    stop_on_success: bool = True
    seed: int = 7
    device: str | None = None
    out_dir: pathlib.Path = pathlib.Path("data/libero/future_visuotactile/pi05_vt_spatial_30000")


def main(args: Args) -> None:
    _validate_args(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    _patch_torch_load_for_libero_init_states()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, input_transform, output_transform = _load_model_and_transforms(args, device)

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    if not 0 <= args.task_id < task_suite.n_tasks:
        raise ValueError(f"Task id {args.task_id} is outside [0, {task_suite.n_tasks}).")
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    if args.num_episodes > len(initial_states):
        raise ValueError(
            f"Requested {args.num_episodes} episodes, but task {args.task_id} only has "
            f"{len(initial_states)} initial states."
        )
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

    logging.info("Task: %s", task_description)
    logging.info("Writing predictions to %s", args.out_dir.resolve())

    try:
        for episode_idx in range(args.num_episodes):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

            element, head_img = _prepare_policy_input(obs, task_description, args.resize_size)
            model_inputs = input_transform(element)
            torch_inputs = _to_torch_batch(model_inputs, device)
            observation = _model.Observation.from_dict(torch_inputs)

            with torch.inference_mode():
                outputs = model.sample_actions(
                    device,
                    observation,
                    num_steps=args.sample_steps,
                    return_future_visuotactile=True,
                )

            normalized_actions = outputs["actions"][0].detach().cpu().numpy()
            normalized_state = torch_inputs["state"][0].detach().cpu().numpy()
            actions = output_transform({"state": normalized_state, "actions": normalized_actions})["actions"]
            pred = outputs["future_visuotactile"][0].detach().cpu().numpy()
            pred_images = _decode_future_images(pred)
            rollout_steps = _resolve_rollout_steps(args.rollout_steps, actions, pred_images)
            head_img = _resize_to_match(head_img, pred_images[0])

            actual_images = []
            executed_actions = []
            rewards = []
            success = False
            for step_idx in range(rollout_steps):
                obs, reward, done, _ = env.step(actions[step_idx].tolist())
                actual_image = _prepare_head_image(obs, args.resize_size)
                actual_images.append(_resize_to_match(actual_image, pred_images[step_idx]))
                executed_actions.append(actions[step_idx])
                rewards.append(float(reward))
                success = success or bool(done)
                if done and args.stop_on_success:
                    break

            episode_dir = args.out_dir / f"episode_{episode_idx:02d}"
            _write_episode_outputs(
                episode_dir,
                head_img,
                pred_images,
                actual_images,
                np.asarray(executed_actions),
                args.video_fps,
                {
                    "task_suite": args.task_suite_name,
                    "task_id": args.task_id,
                    "task_description": str(task_description),
                    "episode": episode_idx,
                    "requested_rollout_steps": rollout_steps,
                    "executed_rollout_steps": len(actual_images),
                    "generated_future_steps": len(pred_images),
                    "success": success,
                    "rewards": rewards,
                },
            )
            logging.info(
                "Episode %d: executed %d predicted actions (success=%s); saved comparison to %s",
                episode_idx,
                len(actual_images),
                success,
                episode_dir / "comparison.mp4",
            )
    finally:
        env.close()


def _load_model_and_transforms(args: Args, device: str):
    train_config = training_config.get_config(args.config_name)
    train_config = dataclasses.replace(
        train_config,
        model=dataclasses.replace(train_config.model, pytorch_compile_mode=None),
    )

    weight_path = args.checkpoint_dir / "model.safetensors"
    model = train_config.model.load_pytorch(train_config, str(weight_path))
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model = model.to(device)
    model.eval()

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("LIBERO data config must define an asset_id for normalization stats.")
    norm_stats = checkpoints.load_norm_stats(args.checkpoint_dir / "assets", data_config.asset_id)
    input_transform = transforms.compose(
        [
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )
    output_transform = transforms.compose(
        [
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]
    )
    return model, input_transform, output_transform


def _validate_args(args: Args) -> None:
    weight_path = args.checkpoint_dir / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weight_path}")
    if args.num_episodes < 1:
        raise ValueError("num_episodes must be at least 1.")
    if args.num_steps_wait < 0:
        raise ValueError("num_steps_wait must be non-negative.")
    if args.sample_steps < 1:
        raise ValueError("sample_steps must be at least 1.")
    if args.rollout_steps is not None and args.rollout_steps < 1:
        raise ValueError("rollout_steps must be at least 1 when set.")
    if args.video_fps < 1:
        raise ValueError("video_fps must be at least 1.")


def _patch_torch_load_for_libero_init_states() -> None:
    original_load = torch.load

    def load_with_legacy_default(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_with_legacy_default


def _to_torch_batch(value, device: str):
    if isinstance(value, dict):
        return {k: _to_torch_batch(v, device) for k, v in value.items()}
    array = np.array(value, copy=True)
    return torch.from_numpy(array).to(device)[None, ...]


def _prepare_policy_input(obs: dict, task_description: str, resize_size: int) -> tuple[dict, np.ndarray]:
    head_img = _prepare_head_image(obs, resize_size)
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    wrist_img = _convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )
    return (
        {
            "observation/image": head_img,
            "observation/wrist_image": wrist_img,
            "observation/state": state,
            "prompt": str(task_description),
        },
        head_img,
    )


def _prepare_head_image(obs: dict, resize_size: int) -> np.ndarray:
    head_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    return _convert_to_uint8(image_tools.resize_with_pad(head_img, resize_size, resize_size))


def _decode_future_images(pred: np.ndarray) -> list[np.ndarray]:
    pred = np.asarray(pred)
    if pred.ndim != 4:
        raise ValueError(f"Expected [horizon, H, W, C] or [horizon, C, H, W], got {pred.shape}.")
    if pred.shape[1] in (1, 3):
        pred = np.transpose(pred, (0, 2, 3, 1))
    if pred.shape[-1] not in (1, 3):
        raise ValueError(f"Expected 1 or 3 image channels, got {pred.shape}.")
    pred = np.clip((pred + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    if pred.shape[-1] == 1:
        pred = np.repeat(pred, 3, axis=-1)
    return list(pred)


def _resolve_rollout_steps(
    requested_steps: int | None,
    actions: np.ndarray,
    pred_images: list[np.ndarray],
) -> int:
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected unnormalized LIBERO actions with shape [horizon, 7], got {actions.shape}.")
    if not np.isfinite(actions).all():
        raise ValueError("Predicted actions contain NaN or infinity.")
    available_steps = min(len(actions), len(pred_images))
    if available_steps < 1:
        raise ValueError("The model did not return any aligned actions and future frames.")
    if requested_steps is not None and requested_steps > available_steps:
        raise ValueError(
            f"rollout_steps={requested_steps}, but only {available_steps} aligned actions/future frames are available."
        )
    return available_steps if requested_steps is None else requested_steps


def _convert_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _resize_to_match(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image.shape == reference.shape:
        return image
    if reference.ndim != 3 or reference.shape[-1] != 3:
        raise ValueError(f"Expected an RGB reference image, got {reference.shape}.")
    return _convert_to_uint8(image_tools.resize_with_pad(image, reference.shape[0], reference.shape[1]))


def _write_episode_outputs(
    episode_dir: pathlib.Path,
    initial_image: np.ndarray,
    pred_images: list[np.ndarray],
    actual_images: list[np.ndarray],
    actions: np.ndarray,
    fps: int,
    metadata: dict,
) -> None:
    if len(actual_images) > len(pred_images):
        raise ValueError(
            f"Actual frame count cannot exceed predicted frame count, got {len(actual_images)} and {len(pred_images)}."
        )
    if not pred_images or not actual_images:
        raise ValueError("At least one predicted/actual frame pair is required.")

    episode_dir.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(episode_dir / "initial.png", initial_image)
    for idx, image in enumerate(pred_images):
        imageio.imwrite(episode_dir / f"pred_{idx:02d}.png", image)
    for idx, image in enumerate(actual_images):
        imageio.imwrite(episode_dir / f"actual_{idx:02d}.png", image)

    aligned_pred_images = pred_images[: len(actual_images)]
    comparison_frames = [
        _make_comparison_frame(prediction, actual, frame_idx + 1)
        for frame_idx, (prediction, actual) in enumerate(zip(aligned_pred_images, actual_images, strict=True))
    ]
    imageio.mimwrite(episode_dir / "pred_future.mp4", pred_images, fps=fps)
    imageio.mimwrite(episode_dir / "actual_rollout.mp4", [initial_image, *actual_images], fps=fps)
    imageio.mimwrite(episode_dir / "comparison.mp4", comparison_frames, fps=fps)
    imageio.imwrite(episode_dir / "contact_sheet.png", np.concatenate([initial_image, *pred_images], axis=1))
    imageio.imwrite(episode_dir / "comparison_contact_sheet.png", np.concatenate(comparison_frames, axis=0))
    np.save(episode_dir / "executed_actions.npy", actions)

    metadata = {**metadata, "image_metrics": _compute_image_metrics(aligned_pred_images, actual_images)}
    with (episode_dir / "metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2)


def _make_comparison_frame(prediction: np.ndarray, actual: np.ndarray, timestep: int) -> np.ndarray:
    if prediction.shape != actual.shape:
        raise ValueError(f"Prediction and actual image shapes must match, got {prediction.shape} and {actual.shape}.")
    error = np.abs(prediction.astype(np.int16) - actual.astype(np.int16))
    error = np.clip(error.astype(np.float32) * 4.0, 0, 255).astype(np.uint8)
    height, width = prediction.shape[:2]
    header_height = 32
    canvas = Image.new("RGB", (3 * width, height + header_height), "white")
    draw = ImageDraw.Draw(canvas)
    panels = (
        (f"Prediction t+{timestep}", prediction),
        (f"Environment t+{timestep}", actual),
        ("|error| x4", error),
    )
    for column, (label, image) in enumerate(panels):
        x = column * width
        canvas.paste(Image.fromarray(image), (x, header_height))
        draw.text((x + 4, 9), label, fill="black")
    return np.asarray(canvas)


def _compute_image_metrics(
    pred_images: list[np.ndarray],
    actual_images: list[np.ndarray],
) -> dict:
    prediction = np.asarray(pred_images, dtype=np.float32) / 255.0
    actual = np.asarray(actual_images, dtype=np.float32) / 255.0
    squared_error = (prediction - actual) ** 2
    absolute_error = np.abs(prediction - actual)
    frame_mse = squared_error.mean(axis=(1, 2, 3))
    frame_mae = absolute_error.mean(axis=(1, 2, 3))
    frame_psnr = -10.0 * np.log10(np.maximum(frame_mse, 1e-12))
    overall_mse = float(squared_error.mean())
    return {
        "mse": overall_mse,
        "mae": float(absolute_error.mean()),
        "psnr_db": float(-10.0 * math.log10(max(overall_mse, 1e-12))),
        "per_frame": [
            {
                "timestep": index + 1,
                "mse": float(mse),
                "mae": float(mae),
                "psnr_db": float(psnr),
            }
            for index, (mse, mae, psnr) in enumerate(zip(frame_mse, frame_mae, frame_psnr, strict=True))
        ],
    }


def _get_libero_env(task, resolution: int, seed: int):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


def _quat2axisangle(quat):
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
