from pathlib import Path
import random
import time
import argparse
from itertools import chain

import torch
import genesis as gs

from helpers import (
    build_urdf,
    build_random_tree,
    init_genesis,
    get_cpu_mem_mb,
    get_gpu_mem_mb,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_DIR = PROJECT_ROOT / "urdf"

DT = 0.01
WARMUP_STEPS = 500
SIM_STEPS = 3000


def normalize_device(device: str) -> str:
    return "cpu" if device.lower() == "cpu" else "cuda"


def create_robots(scene, num_robots: int, robot_size: int, urdf_path: str | None):
    robots = []

    for i in range(num_robots):
        pos = random.choice(
            [(0.5 * (i - 2), 0, 0), (0, 0.5 * (i - 2), 0)]
        )

        if urdf_path is not None:
            path = Path(urdf_path)
            if not path.is_absolute():
                path = (PROJECT_ROOT / path).resolve()

            if path.suffix == ".xacro":
                build_urdf(path)
                urdf_file = path.with_suffix(".urdf")
            else:
                urdf_file = path

            robot = scene.add_entity(
                gs.morphs.URDF(file=str(urdf_file), pos=pos)
            )
        else:
            xacro_file = URDF_DIR / f"random_robot_{i}.urdf.xacro"
            build_random_tree(
                num_modules=robot_size,
                robot_idx=i,
                output_path=str(xacro_file),
                robot_line=False,
            )
            build_urdf(xacro_file)
            robot = scene.add_entity(
                gs.morphs.URDF(file=str(xacro_file.with_suffix("")), pos=pos)
            )

        robots.append(robot)

    return robots


def simulate(
    gen_size: int,
    num_modules: int,
    scene,
    device: str,
    urdf_path: str | None,
):
    device = normalize_device(device)
    torch_device = torch.device(device)

    robots = create_robots(scene, gen_size, num_modules, urdf_path)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    sim_cpu_start = get_cpu_mem_mb()
    sim_gpu_start = get_gpu_mem_mb()

    scene.build()

    initial_positions = [robot.get_pos() for robot in robots]
    control_params = []

    for robot in robots:
        dofs_idx = list(
            chain.from_iterable(
                joint.dofs_idx_local
                for joint in robot.joints
                if "module" in joint.name
            )
        )
        robot.dofs_idx = dofs_idx
        n = len(dofs_idx)

        if n == 0:
            control_params.append(
                {
                    "phases": torch.empty(0, device=torch_device),
                    "amplitudes": torch.empty(0, device=torch_device),
                    "frequencies": torch.empty(0, device=torch_device),
                }
            )
            continue

        robot.set_dofs_kp(torch.full((n,), 80.0, device=torch_device), dofs_idx)
        robot.set_dofs_kv(torch.full((n,), 2.0, device=torch_device), dofs_idx)
        robot.set_dofs_force_range(
            torch.full((n,), -3.0, device=torch_device),
            torch.full((n,), 3.0, device=torch_device),
            dofs_idx,
        )

        control_params.append(
            {
                "phases": torch.rand(n, device=torch_device) * (2 * torch.pi),
                "amplitudes": 0.4 + 0.2 * torch.rand(n, device=torch_device),
                "frequencies": 0.8 + 0.4 * torch.rand(n, device=torch_device),
            }
        )

    t = 0.0

    def step():
        nonlocal t
        t += DT
        for robot, params in zip(robots, control_params):
            if not robot.dofs_idx:
                continue
            targets = params["amplitudes"] * torch.sin(
                2 * torch.pi * params["frequencies"] * t + params["phases"]
            )
            robot.control_dofs_position(targets, robot.dofs_idx)
        scene.step()

    for _ in range(WARMUP_STEPS):
        step()

    start_time = time.perf_counter()

    for _ in range(SIM_STEPS):
        step()

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed_time = time.perf_counter() - start_time

    final_positions = [robot.get_pos() for robot in robots]
    distances = torch.norm(
        torch.stack(final_positions) - torch.stack(initial_positions),
        dim=1,
    )
    print("Distances:", distances.cpu().tolist())

    if urdf_path is None:
        for i in range(len(robots)):
            (URDF_DIR / f"random_robot_{i}.urdf.xacro").unlink(missing_ok=True)
            (URDF_DIR / f"random_robot_{i}.urdf").unlink(missing_ok=True)

    end_cpu_mem = get_cpu_mem_mb()
    end_gpu_mem = get_gpu_mem_mb()

    return {
        "fps": SIM_STEPS / elapsed_time,
        "steps": SIM_STEPS,
        "elapsed_time": elapsed_time,
        "device": device,
        "cpu_mem_mb": end_cpu_mem,
        "gpu_mem_mb": end_gpu_mem,
        "simulation_ram_mb": end_cpu_mem - sim_cpu_start,
        "simulation_vram_mb": end_gpu_mem - sim_gpu_start,
    }


def main(num_robots: int, num_modules: int, device: str, urdf_path: str | None):
    device = normalize_device(device)

    start_cpu_mem = get_cpu_mem_mb()
    start_gpu_mem = get_gpu_mem_mb()

    gs.init(backend=gs.cpu if device == "cpu" else gs.gpu)
    scene = init_genesis(viewer=True)

    results = simulate(num_robots, num_modules, scene, device, urdf_path)

    cpu_delta = results["cpu_mem_mb"] - start_cpu_mem
    gpu_delta = results["gpu_mem_mb"] - start_gpu_mem if device == "cuda" else 0.0

    print("\nBenchmark Results")
    header = (
        f"{'Device':<8}"
        f"{'Time (s)':>12}"
        f"{'FPS':>12}"
        f"{'Sim RAM (MB)':>15}"
        f"{'Sim VRAM (MB)':>15}"
        f"{'Total RAM (MB)':>15}"
        f"{'Total VRAM (MB)':>15}"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{device:<8}"
        f"{results['elapsed_time']:>12.2f}"
        f"{results['fps']:>12.2f}"
        f"{results['simulation_ram_mb']:>15.2f}"
        f"{(results['simulation_vram_mb'] if device == 'cuda' else 0.0):>15.2f}"
        f"{cpu_delta:>15.2f}"
        f"{gpu_delta:>15.2f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genesis Modular Robot Simulation")
    parser.add_argument("-r", "--robots", type=int, default=1)
    parser.add_argument("-m", "--modules", type=int, default=5)
    parser.add_argument("--urdf", type=str, default=None)
    parser.add_argument("-d", "--device", default="cpu")

    args = parser.parse_args()
    main(args.robots, args.modules, args.device, args.urdf)
