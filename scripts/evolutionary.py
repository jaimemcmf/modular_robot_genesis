import argparse
import math
import random
import time
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import genesis as gs
import torch

from helpers import (
    build_urdf,
    build_random_tree,
    init_genesis,
    get_cpu_mem_mb,
    get_gpu_mem_mb,
)

DT = 0.01
EVAL_STEPS = 2000
TOTAL_SIM_STEPS = 0
TOTAL_SIM_TIME = 0.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_DIR = PROJECT_ROOT / "urdf"

TOTAL_SIM_STEPS = 0

@dataclass
class Robot:
    id: int
    num_modules: int
    entity: any = None
    fitness: float = 0.0
    phases: torch.Tensor | None = None
    amplitudes: torch.Tensor | None = None
    frequencies: torch.Tensor | None = None

def add_gen_to_scene(scene, robots):
    for i, robot in enumerate(robots):
        xacro = URDF_DIR / f"random_robot_{robot.id}.urdf.xacro"
        build_random_tree(robot.num_modules, i, str(xacro), robot_line=False)
        build_urdf(xacro)

        pos = random.choice([(0.5 * (i - 2), 0, 0), (0, 0.5 * (i - 2), 0)])
        robot.entity = scene.add_entity(
            gs.morphs.URDF(file=str(xacro.with_suffix("")), pos=pos)
        )

def initialization(gen_size: int, device: torch.device):
    robots = []

    for i in range(gen_size):
        n = random.randint(1, 5)
        robots.append(
            Robot(
                id=i,
                num_modules=n,
                phases=torch.empty(n, device=device).uniform_(0.0, 2 * math.pi),
                amplitudes=torch.empty(n, device=device).uniform_(0.4, 0.6),
                frequencies=torch.empty(n, device=device).uniform_(0.8, 1.2),
            )
        )

    return robots


def evaluate(scene, robots, device: torch.device):
    global TOTAL_SIM_STEPS, TOTAL_SIM_TIME

    initial_positions = torch.stack([r.entity.get_pos() for r in robots])

    control_params = []

    for robot in robots:
        dofs_idx = list(
            chain.from_iterable(
                j.dofs_idx_local for j in robot.entity.joints if "module" in j.name
            )
        )
        robot.entity.dofs_idx = dofs_idx
        n = len(dofs_idx)

        if n == 0:
            control_params.append(None)
            continue

        robot.entity.set_dofs_kp(torch.full((n,), 80.0, device=device), dofs_idx)
        robot.entity.set_dofs_kv(torch.full((n,), 2.0, device=device), dofs_idx)
        robot.entity.set_dofs_force_range(
            torch.full((n,), -3.0, device=device),
            torch.full((n,), 3.0, device=device),
            dofs_idx,
        )

        control_params.append(
            (robot.phases, robot.amplitudes, robot.frequencies)
        )

    t = 0.0
    sim_start = time.perf_counter()
    for _ in range(EVAL_STEPS):
        t += DT
        for robot, params in zip(robots, control_params):
            if len(robot.entity.dofs_idx) == 0:
                continue

            phases, amplitudes, frequencies = params

            targets = amplitudes * torch.sin(
                2 * torch.pi * frequencies * t + phases
            )
            robot.entity.control_dofs_position(targets, robot.entity.dofs_idx)
        scene.step()
        TOTAL_SIM_STEPS += 1
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    TOTAL_SIM_TIME += time.perf_counter() - sim_start

    final_positions = torch.stack([r.entity.get_pos() for r in robots])
    distances = torch.norm(final_positions - initial_positions, dim=1)

    for robot, d in zip(robots, distances):
        robot.fitness = float(d.item())

    print("Distances:", [r.fitness for r in robots])

    for robot in robots:
        (URDF_DIR / f"random_robot_{robot.id}.urdf.xacro").unlink(missing_ok=True)
        (URDF_DIR / f"random_robot_{robot.id}.urdf").unlink(missing_ok=True)


def selection(robots):
    return sorted(robots, key=lambda r: r.fitness, reverse=True)[: len(robots) // 2]


def mutation(robots):

    p_num_modules = 0.3
    p_phase = 0.15
    p_amplitude = 0.15
    p_frequency = 0.15

    phase_sigma = 0.2
    amplitude_scale_sigma = 0.2
    frequency_scale_sigma = 0.2

    for r in robots:

        if random.random() < p_num_modules:
            change = random.choice([-1, 1])
            new_n = max(1, r.num_modules + change)

            if new_n > r.num_modules:
                r.phases = torch.cat(
                    [r.phases, r.phases[-1:] + torch.randn(1, device=r.phases.device) * phase_sigma]
                )
                r.amplitudes = torch.cat(
                    [r.amplitudes, r.amplitudes[-1:] * (1 + torch.randn(1) * amplitude_scale_sigma)]
                )
                r.frequencies = torch.cat(
                    [r.frequencies, r.frequencies[-1:] * (1 + torch.randn(1) * frequency_scale_sigma)]
                )

            r.num_modules = new_n

        n = r.num_modules
        r.phases = r.phases[:n]
        r.amplitudes = r.amplitudes[:n]
        r.frequencies = r.frequencies[:n]

        if r.phases.numel() < n:
            r.phases = torch.cat([r.phases, torch.zeros(n - r.phases.numel(), device=r.phases.device)])
            r.amplitudes = torch.cat([r.amplitudes, torch.zeros(n - r.amplitudes.numel(), device=r.amplitudes.device)])
            r.frequencies = torch.cat([r.frequencies, torch.zeros(n - r.frequencies.numel(), device=r.frequencies.device)])

        if n > 0:
            mask = torch.rand(n, device=r.phases.device) < p_phase
            r.phases = torch.where(
                mask,
                r.phases + torch.randn(n, device=r.phases.device) * phase_sigma,
                r.phases,
            )

        if n > 0:
            mask = torch.rand(n, device=r.amplitudes.device) < p_amplitude
            scale = 1 + torch.randn(n, device=r.amplitudes.device) * amplitude_scale_sigma
            r.amplitudes = torch.where(
                mask,
                torch.clamp(r.amplitudes * scale, 0.0, 1.0),
                r.amplitudes,
            )

        if n > 0:
            mask = torch.rand(n, device=r.frequencies.device) < p_frequency
            scale = 1 + torch.randn(n, device=r.frequencies.device) * frequency_scale_sigma
            r.frequencies = torch.where(
                mask,
                torch.clamp(r.frequencies * scale, 0.1, 5.0),
                r.frequencies,
            )

        r.fitness = 0.0

    return robots

def crossover(parents, gen_size):
    children = []

    for _ in range(gen_size // 2):
        p1, p2 = random.sample(parents, 2)
        n = random.choice([p1.num_modules, p2.num_modules])

        def mix(a: torch.Tensor, b: torch.Tensor, n: int):
            device = a.device

            if a.numel() < n:
                a = torch.cat([a, torch.zeros(n - a.numel(), device=device)])
            else:
                a = a[:n]

            if b.numel() < n:
                b = torch.cat([b, torch.zeros(n - b.numel(), device=device)])
            else:
                b = b[:n]

            mask = torch.rand(n, device=device) < 0.5
            return torch.where(mask, a, b)

        for _ in range(2):
            children.append(
                Robot(
                    id=-1,
                    num_modules=n,
                    phases=mix(p1.phases, p2.phases, n),
                    amplitudes=mix(p1.amplitudes, p2.amplitudes, n),
                    frequencies=mix(p1.frequencies, p2.frequencies, n),
                )
            )

    for i, c in enumerate(children):
        c.id = i

    return children

def main(num_generations, gen_size, device: str):

    device = "cpu" if device.lower() == "cpu" else "cuda"
    torch_device = torch.device(device)

    start_time = time.perf_counter()
    start_cpu = get_cpu_mem_mb()
    start_gpu = get_gpu_mem_mb()

    gs.init(backend=gs.cpu if device == "cpu" else gs.gpu)

    scene = init_genesis()
    robots = initialization(gen_size, torch_device)
    add_gen_to_scene(scene, robots)
    scene.build()
    evaluate(scene, robots, torch_device)

    for g in range(num_generations):
        robots = mutation(crossover(selection(robots), gen_size))
        scene = init_genesis()
        add_gen_to_scene(scene, robots)
        scene.build()
        evaluate(scene, robots, torch_device)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    sim_fps = TOTAL_SIM_STEPS / TOTAL_SIM_TIME if TOTAL_SIM_TIME > 0 else 0.0

    print("\n===== BENCHMARK SUMMARY =====")
    print(f"Device         : {device}")
    print(f"Wall time (s)  : {elapsed:.2f}")
    print(f"Sim time (s)        : {TOTAL_SIM_TIME:.2f}")
    print(f"Total sim steps     : {TOTAL_SIM_STEPS}")
    print(f"Simulation FPS      : {sim_fps:.2f}")
    print(f"CPU RAM Δ (MB): {get_cpu_mem_mb() - start_cpu:.2f}")
    if device == "cuda":
        print(f"GPU VRAM Δ (MB): {get_gpu_mem_mb() - start_gpu:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--generations", type=int, default=3)
    parser.add_argument("-n", "--gen-size", type=int, default=4)
    parser.add_argument("-d", "--device", default="cpu")
    args = parser.parse_args()

    main(args.generations, args.gen_size, args.device)
