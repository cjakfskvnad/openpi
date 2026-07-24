import dataclasses
import logging
import math
import pathlib

import imageio.v2 as imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import torch
import tyro

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import libero_policy
from openpi.shared import image_tools
from openpi.training import checkpoints
from openpi.training import config as training_config

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    checkpoint_dir: pathlib.Path = pathlib.Path(
        "checkpoints/pi05_visuotactile_libero/libero_pi05_vt_spatial_gpu1234_bz32/30000"
    )
    config_name: str = "pi05_visuotactile_libero"
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    num_episodes: int = 3
    num_steps_wait: int = 10
    resize_size: int = 224
    sample_steps: int = 10
    seed: int = 7
    device: str | None = None
    out_dir: pathlib.Path = pathlib.Path("data/libero/future_visuotactile/pi05_vt_spatial_30000")


def main(args: Args) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    _patch_torch_load_for_libero_init_states()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, input_transform = _load_model_and_transforms(args, device)

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

    logging.info("Task: %s", task_description)
    logging.info("Writing predictions to %s", args.out_dir.resolve())

    try:
        for episode_idx in range(args.num_episodes):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

            head_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            head_img = _convert_to_uint8(image_tools.resize_with_pad(head_img, args.resize_size, args.resize_size))
            wrist_img = _convert_to_uint8(image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))
            state = np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            )

            element = {
                "observation/image": head_img,
                "observation/wrist_image": wrist_img,
                "observation/state": state,
                "prompt": str(task_description),
            }
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

            pred = outputs["future_visuotactile"][0].detach().cpu().numpy()
            pred_images = _decode_future_images(pred)
            episode_dir = args.out_dir / f"episode_{episode_idx:02d}"
            _write_episode_outputs(episode_dir, head_img, pred_images)
            logging.info("Episode %d: saved %d predicted frames to %s", episode_idx, len(pred_images), episode_dir)
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
            libero_policy.LiberoVisuoTactileInputs(model_type=train_config.model.model_type),
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )
    return model, input_transform


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


def _decode_future_images(pred: np.ndarray) -> list[np.ndarray]:
    pred = np.asarray(pred)
    if pred.ndim != 4:
        raise ValueError(f"Expected [horizon, H, W, C] or [horizon, C, H, W], got {pred.shape}.")
    if pred.shape[1] in (1, 3):
        pred = np.transpose(pred, (0, 2, 3, 1))
    pred = np.clip((pred + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    return list(pred)


def _convert_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _write_episode_outputs(episode_dir: pathlib.Path, initial_image: np.ndarray, pred_images: list[np.ndarray]) -> None:
    episode_dir.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(episode_dir / "initial.png", initial_image)
    for idx, image in enumerate(pred_images):
        imageio.imwrite(episode_dir / f"pred_{idx:02d}.png", image)
    imageio.mimwrite(episode_dir / "pred_future.mp4", pred_images, fps=5)
    imageio.imwrite(episode_dir / "contact_sheet.png", np.concatenate([initial_image, *pred_images], axis=1))


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
