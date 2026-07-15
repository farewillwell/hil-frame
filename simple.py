#!/usr/bin/env python3
"""
持续显示 LIBERO 的主视角（agentview）。

功能：
- 使用 LIBERO OffScreenRenderEnv 持续生成 agentview 图像
- 通过 OpenCV 在 WSLg / Linux 桌面实时显示
- q 或 Esc：退出
- r：重置环境
- 空格：暂停 / 继续
python man.py \
  --suite libero_goal \
  --task-id 0 \
  --width 512 \
  --height 512 \
  --fps 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import cv2
import numpy as np

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


VALID_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="持续显示 LIBERO 环境的 agentview 主视角。"
    )
    parser.add_argument("--suite", default="libero_goal", choices=VALID_SUITES)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--task-order-index", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--gripper-action", type=float, default=0.0)
    return parser.parse_args()


def get_image(obs: dict[str, Any], camera_name: str) -> np.ndarray:
    key = f"{camera_name}_image"
    if key not in obs:
        available = sorted(k for k in obs if k.endswith("_image"))
        raise KeyError(f"观测中不存在 {key!r}；可用图像键：{available}")

    image = np.asarray(obs[key])

    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"{key} 形状异常：{image.shape}")

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.max(initial=0.0) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.shape[-1] == 4:
        image = image[..., :3]

    return image


def draw_overlay(
    frame_bgr: np.ndarray,
    instruction: str,
    step: int,
    measured_fps: float,
    success: bool,
    paused: bool,
) -> np.ndarray:
    frame = frame_bgr.copy()
    status = "PAUSED" if paused else ("SUCCESS" if success else "RUNNING")
    lines = [
        f"Task: {instruction}",
        f"Step: {step}    FPS: {measured_fps:.1f}    Status: {status}",
        "Keys: q/Esc quit | r reset | Space pause",
    ]

    overlay = frame.copy()
    bar_height = min(frame.shape[0], 92)
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], bar_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0.0, frame)

    y = 24
    for text in lines:
        max_chars = max(20, frame.shape[1] // 9)
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 27

    return frame


def main() -> int:
    args = parse_args()

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite](
        task_order_index=args.task_order_index
    )

    task_count = task_suite.get_num_tasks()
    if not 0 <= args.task_id < task_count:
        raise ValueError(
            f"--task-id={args.task_id} 超出范围；{args.suite} 共 {task_count} 个任务"
        )
    if args.fps <= 0 or args.control_freq <= 0:
        raise ValueError("--fps 和 --control-freq 必须大于 0")

    task = task_suite.get_task(args.task_id)
    bddl_file = task_suite.get_task_bddl_file_path(args.task_id)
    init_states = task_suite.get_task_init_states(args.task_id)
    if len(init_states) == 0:
        raise RuntimeError("该任务没有初始状态")

    init_state_id = args.init_state_id % len(init_states)

    print("=" * 72)
    print(f"Suite       : {args.suite}")
    print(f"Task ID     : {args.task_id}")
    print(f"Instruction : {task.language}")
    print(f"BDDL        : {bddl_file}")
    print(f"Init state  : {init_state_id}/{len(init_states) - 1}")
    print(f"DISPLAY     : {os.environ.get('DISPLAY', '<unset>')}")
    print(f"WAYLAND     : {os.environ.get('WAYLAND_DISPLAY', '<unset>')}")
    print("Keys        : q/Esc quit | r reset | Space pause")
    print("=" * 72)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_names=[args.camera],
        camera_heights=args.height,
        camera_widths=args.width,
        control_freq=args.control_freq,
        horizon=args.max_steps,
        ignore_done=False,
        use_camera_obs=True,
    )
    env.seed(args.seed)

    window_name = f"LIBERO - {args.suite} - task {args.task_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    idle_action = np.zeros(7, dtype=np.float32)
    idle_action[-1] = np.clip(args.gripper_action, -1.0, 1.0)

    step_count = 0
    paused = False
    success = False
    measured_fps = 0.0
    fps_counter = 0
    fps_window_start = time.perf_counter()
    target_period = 1.0 / args.fps
    next_tick = time.perf_counter()

    def reset_episode() -> dict[str, Any]:
        nonlocal step_count, success, next_tick
        env.reset()
        obs = env.set_init_state(init_states[init_state_id])
        step_count = 0
        success = False
        next_tick = time.perf_counter()
        return obs

    obs = reset_episode()

    try:
        while True:
            pending_reset = False

            if not paused:
                obs, reward, done, _ = env.step(idle_action)
                step_count += 1
                success = bool(env.check_success())

                if success or done or step_count >= args.max_steps:
                    pending_reset = True
                    reason = "success" if success else ("done" if done else "max_steps")
                    print(
                        f"[reset] reason={reason}, steps={step_count}, reward={reward}"
                    )

            image = get_image(obs, args.camera)
            image = np.flipud(image)

            frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            fps_counter += 1
            now = time.perf_counter()
            elapsed = now - fps_window_start
            if elapsed >= 0.5:
                measured_fps = fps_counter / elapsed
                fps_counter = 0
                fps_window_start = now

            if not args.no_overlay:
                frame_bgr = draw_overlay(
                    frame_bgr,
                    task.language,
                    step_count,
                    measured_fps,
                    success,
                    paused,
                )

            cv2.imshow(window_name, frame_bgr)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                obs = reset_episode()
                print("[reset] manual")
                continue
            if key == ord(" "):
                paused = not paused
                print(f"[pause] {paused}")
                next_tick = time.perf_counter()
                continue

            if pending_reset:
                cv2.waitKey(150)
                obs = reset_episode()
                continue

            next_tick += target_period
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif -sleep_time > target_period:
                next_tick = time.perf_counter()

    except KeyboardInterrupt:
        print("\n[exit] interrupted")
    finally:
        env.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[fatal] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
