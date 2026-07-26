from __future__ import annotations

import collections
import dataclasses
import logging
import math
import multiprocessing
import pathlib
import traceback

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    num_parallel_envs: int = 1  # Number of simulator processes and fixed model batch size.

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    task_id: int | None = None  # If set, evaluate only this task in the suite.
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    save_video: bool = True  # Save replay videos for each rollout.

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    if args.num_parallel_envs < 1:
        raise ValueError("num_parallel_envs must be at least 1.")
    if args.num_trials_per_task < 1:
        raise ValueError("num_trials_per_task must be at least 1.")
    if args.replan_steps < 1:
        raise ValueError("replan_steps must be at least 1.")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    if args.task_id is not None and not 0 <= args.task_id < num_tasks_in_suite:
        raise ValueError(f"Task id {args.task_id} is outside [0, {num_tasks_in_suite}).")
    task_ids = range(num_tasks_in_suite) if args.task_id is None else [args.task_id]
    for task_id in tqdm.tqdm(task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        if args.num_trials_per_task > len(initial_states):
            raise ValueError(
                f"Requested {args.num_trials_per_task} trials for task {task_id}, "
                f"but only {len(initial_states)} initial states are available."
            )

        task_description = task.language
        task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        num_workers = min(args.num_parallel_envs, args.num_trials_per_task)
        workers = [
            _SimWorker(task_bddl_file, LIBERO_ENV_RESOLUTION, args.seed, worker_id) for worker_id in range(num_workers)
        ]

        task_episodes, task_successes = 0, 0
        next_episode_idx = 0
        slots = []
        try:
            for worker in workers:
                worker.wait_until_ready()

            # Initial resets are dispatched before receiving any result, so simulator
            # initialization work happens concurrently.
            pending_resets = []
            for worker in workers:
                episode_idx = next_episode_idx
                next_episode_idx += 1
                worker.send_reset(initial_states[episode_idx])
                pending_resets.append((worker, episode_idx))
            for worker, episode_idx in pending_resets:
                slots.append(_EpisodeSlot(worker=worker, episode_idx=episode_idx, obs=worker.recv_reset()))

            with tqdm.tqdm(total=args.num_trials_per_task, desc=f"Task {task_id}") as episode_progress:
                while slots:
                    ready_slots = []
                    ready_elements = []
                    for slot in slots:
                        if slot.t < args.num_steps_wait:
                            continue

                        element, replay_image = _prepare_policy_input(slot.obs, task_description, args.resize_size)
                        slot.replay_images.append(replay_image)
                        if not slot.action_plan:
                            ready_slots.append(slot)
                            ready_elements.append(element)

                    if ready_slots:
                        # Keep the model batch shape fixed to avoid a new JAX compilation
                        # whenever replacement episodes become desynchronized.
                        padded_elements = [
                            *ready_elements,
                            *([ready_elements[-1]] * (num_workers - len(ready_elements))),
                        ]
                        batch_results = client.infer_batch(padded_elements)
                        if len(batch_results) != num_workers:
                            raise RuntimeError(f"Expected {num_workers} batch results, got {len(batch_results)}.")
                        for slot, result in zip(ready_slots, batch_results):  # noqa: B905
                            action_chunk = result["actions"]
                            if len(action_chunk) < args.replan_steps:
                                raise ValueError(
                                    f"We want to replan every {args.replan_steps} steps, "
                                    f"but policy only predicts {len(action_chunk)} steps."
                                )
                            slot.action_plan.extend(action_chunk[: args.replan_steps])

                    # Dispatch every simulator step first, then receive every result.
                    # Each OffScreenRenderEnv lives in its own spawned process.
                    dispatched_steps = []
                    for slot in slots:
                        is_wait_step = slot.t < args.num_steps_wait
                        action = LIBERO_DUMMY_ACTION if is_wait_step else slot.action_plan.popleft().tolist()
                        slot.worker.send_step(action)
                        dispatched_steps.append((slot, is_wait_step))

                    finished = []
                    for slot, is_wait_step in dispatched_steps:
                        slot.obs, done = slot.worker.recv_step()
                        slot.t += 1
                        success = bool(done) and not is_wait_step
                        timed_out = slot.t >= max_steps + args.num_steps_wait
                        if success or timed_out:
                            finished.append((slot, success))

                    reset_slots = []
                    retired_slot_ids = set()
                    for slot, success in finished:
                        task_episodes += 1
                        total_episodes += 1
                        if success:
                            task_successes += 1
                            total_successes += 1

                        if args.save_video and slot.replay_images:
                            suffix = "success" if success else "failure"
                            task_segment = task_description.replace(" ", "_")
                            output_name = (
                                f"rollout_task_{task_id:02d}_episode_{slot.episode_idx:03d}_{task_segment}_{suffix}.mp4"
                            )
                            imageio.mimwrite(
                                pathlib.Path(args.video_out_path) / output_name,
                                [np.asarray(x) for x in slot.replay_images],
                                fps=10,
                            )

                        logging.info(f"Task {task_id}, episode {slot.episode_idx}: success={success}")
                        logging.info(f"# episodes completed so far: {total_episodes}")
                        logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
                        episode_progress.update(1)

                        if next_episode_idx < args.num_trials_per_task:
                            slot.episode_idx = next_episode_idx
                            next_episode_idx += 1
                            slot.t = 0
                            slot.action_plan.clear()
                            slot.replay_images.clear()
                            slot.worker.send_reset(initial_states[slot.episode_idx])
                            reset_slots.append(slot)
                        else:
                            retired_slot_ids.add(id(slot))

                    if retired_slot_ids:
                        slots = [slot for slot in slots if id(slot) not in retired_slot_ids]
                    for slot in reset_slots:
                        slot.obs = slot.worker.recv_reset()
        finally:
            for worker in workers:
                worker.close()

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


@dataclasses.dataclass
class _EpisodeSlot:
    worker: _SimWorker
    episode_idx: int
    obs: dict
    t: int = 0
    action_plan: collections.deque = dataclasses.field(default_factory=collections.deque)
    replay_images: list[np.ndarray] = dataclasses.field(default_factory=list)


class _SimWorker:
    """A parent-side handle for one OffScreenRenderEnv process."""

    def __init__(self, task_bddl_file: pathlib.Path, resolution: int, seed: int, worker_id: int):
        context = multiprocessing.get_context("spawn")
        self._connection, child_connection = context.Pipe()
        self._process = context.Process(
            target=_sim_worker_main,
            args=(child_connection, str(task_bddl_file), resolution, seed),
            name=f"libero-sim-{worker_id}",
        )
        self._process.start()
        child_connection.close()

    def wait_until_ready(self) -> None:
        self._recv("ready")

    def send_reset(self, initial_state) -> None:
        self._connection.send(("reset", initial_state))

    def recv_reset(self) -> dict:
        return self._recv("reset")

    def send_step(self, action) -> None:
        self._connection.send(("step", action))

    def recv_step(self) -> tuple[dict, bool]:
        return self._recv("step")

    def close(self) -> None:
        if self._process.is_alive():
            try:
                self._connection.send(("close", None))
                self._recv("closed")
            except (BrokenPipeError, EOFError, OSError, RuntimeError):
                pass
        self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._connection.close()

    def _recv(self, expected_kind: str):
        try:
            kind, payload = self._connection.recv()
        except EOFError as error:
            raise RuntimeError(f"Simulator process exited unexpectedly with code {self._process.exitcode}.") from error
        if kind == "error":
            raise RuntimeError(f"Simulator worker failed:\n{payload}")
        if kind != expected_kind:
            raise RuntimeError(f"Expected simulator response {expected_kind!r}, got {kind!r}.")
        return payload


def _sim_worker_main(connection, task_bddl_file: str, resolution: int, seed: int) -> None:
    env = None
    try:
        env = _get_libero_env(task_bddl_file, resolution, seed)
        connection.send(("ready", None))
        while True:
            command, payload = connection.recv()
            try:
                if command == "reset":
                    env.reset()
                    obs = env.set_init_state(payload)
                    connection.send(("reset", _compact_observation(obs)))
                elif command == "step":
                    obs, _, done, _ = env.step(payload)
                    connection.send(("step", (_compact_observation(obs), bool(done))))
                elif command == "close":
                    connection.send(("closed", None))
                    break
                else:
                    raise ValueError(f"Unknown simulator command: {command!r}")
            except Exception:
                connection.send(("error", traceback.format_exc()))
    except Exception:
        connection.send(("error", traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        connection.close()


def _get_libero_env(task_bddl_file: str, resolution: int, seed: int):
    """Initialize one LIBERO environment inside a simulator worker."""
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env


def _compact_observation(obs) -> dict:
    """Only move fields used by policy inference across the process boundary."""
    return {
        "agentview_image": obs["agentview_image"],
        "robot0_eye_in_hand_image": obs["robot0_eye_in_hand_image"],
        "robot0_eef_pos": obs["robot0_eef_pos"],
        "robot0_eef_quat": obs["robot0_eef_quat"],
        "robot0_gripper_qpos": obs["robot0_gripper_qpos"],
    }


def _prepare_policy_input(obs: dict, task_description: str, resize_size: int) -> tuple[dict, np.ndarray]:
    # Rotate 180 degrees to match training preprocessing.
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))
    element = {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": str(task_description),
    }
    return element, img


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
