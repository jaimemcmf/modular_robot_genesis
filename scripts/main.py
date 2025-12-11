from pathlib import Path
import random
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import math
import genesis as gs
from itertools import chain
import numpy as np
from helpers import build_urdf, build_random_tree, init_genesis
import torch
import argparse
from dataclasses import dataclass

project_root = Path(__file__).resolve().parents[1]
urdf_dir = project_root / "urdf"

def simulate(gen_size: int, num_modules: int, scene):

    robots = create_robots(scene, gen_size, robot_size=num_modules)
    scene.build()

    initial_positions = [robot.get_pos() for robot in robots]

    # Prepare per-robot control params and dof indices
    control_params = []  # list of dicts per robot
    for robot in robots:
        dofs_idx = list(
            chain.from_iterable(
                [
                    joint.dofs_idx_local
                    for joint in robot.joints
                    if "module" in joint.name
                ]
            )
        )
        robot.dofs_idx = dofs_idx
        # Safety: clamp sizes to available DOFs
        n = len(dofs_idx)
        if n == 0:
            control_params.append(
                {"phases": np.array([]), "amplitudes": np.array(
                    []), "frequencies": np.array([])}
            )
            continue

        robot.set_dofs_kp(np.full(n, 80.0), dofs_idx)
        robot.set_dofs_kv(np.full(n, 2.0), dofs_idx)
        robot.set_dofs_force_range(np.full(n, -3.0), np.full(n, 3.0), dofs_idx)

        phases = np.random.uniform(0, 2 * np.pi, n)
        amplitudes = np.random.uniform(0.4, 0.6, n)   # around 0.5
        frequencies = np.random.uniform(0.8, 1.2, n)  # around 1.0 Hz
        control_params.append(
            {"phases": phases, "amplitudes": amplitudes, "frequencies": frequencies}
        )

    t = 0.0
    dt = 0.01
    for _ in range(1000):
        t += dt
        for robot, params in zip(robots, control_params):
            if len(robot.dofs_idx) == 0:
                continue
            targets = params["amplitudes"] * np.sin(
                2 * np.pi * params["frequencies"] * t + params["phases"]
            )
            robot.control_dofs_position(targets, robot.dofs_idx)
        scene.step()

    final_positions = [robot.get_pos() for robot in robots]

    distances = torch.stack(final_positions) - torch.stack(initial_positions)
    distances = torch.norm(distances, dim=1)
    print("Distances:", distances.cpu().tolist())

    for i in range(len(robots)):
        (urdf_dir / f"random_robot_{i}.urdf.xacro").unlink(missing_ok=True)
        (urdf_dir / f"random_robot_{i}.urdf").unlink(missing_ok=True)


def create_robots(scene, num_robots: int, robot_size: int):
    robots = []
    for i in range(num_robots):
        xacro_file = urdf_dir / f"random_robot_{i}.urdf.xacro"
        build_random_tree(
            num_modules=robot_size, robot_idx=i, output_path=str(xacro_file), robot_line=False
        )
        build_urdf(xacro_file)
        possible_locations = [(0.5 * (i - 2), 0, 0), (0, 0.5 * (i - 2), 0)]
        robot = scene.add_entity(
            gs.morphs.URDF(
                file=str(xacro_file.with_suffix("")),
                pos=random.choice(possible_locations),
            )
        )
        robots.append(robot)
    return robots

def main(num_robots: int, num_modules: int):
    gs.init(backend=gs.cpu)
    scene = init_genesis()
    simulate(num_robots, num_modules, scene)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genesis Modular Robot Simulation")

    parser.add_argument(
        "-r", "--robots",
        type=int,
        default=1,
        help="Number of robots to simulate (default = 1)"
    )

    parser.add_argument(
        "-m", "--modules",
        type=int,
        default=5,
        help="Number of modules per robot (default = 5)"
    )

    args = parser.parse_args()
    main(args.robots, args.modules)
    
