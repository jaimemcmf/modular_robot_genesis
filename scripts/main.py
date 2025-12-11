from pathlib import Path
import random
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import math
import genesis as gs
from itertools import chain
import numpy as np
from helpers import build_urdf
import torch
import argparse

project_root = Path(__file__).resolve().parents[1]
urdf_dir = project_root / "urdf"


def simulate(gen_size: int, num_modules: int):

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -3.5, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            res=(960, 640),
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            dt=0.01,
            enable_adjacent_collision=True,
            enable_joint_limit=True,
            gravity=np.array([0.0, -9.81, 0.0]),
            constraint_solver=gs.constraint_solver.Newton,
        ),
    )
    scene.add_entity(gs.morphs.Plane())
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

    # Clean up generated files; ignore if already removed
    for i in range(len(robots)):
        (urdf_dir / f"random_robot_{i}.urdf.xacro").unlink(missing_ok=True)
        (urdf_dir / f"random_robot_{i}.urdf").unlink(missing_ok=True)


def create_robots(scene, gen_size: int, robot_size: int):
    robots = []
    for i in range(gen_size):
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


def build_random_tree(
    num_modules: int, robot_idx="x", output_path: str = "urdf/random_robot.urdf.xacro", robot_line=False
):
    base_half_xy = 0.0747 / 2.0
    module_half_xy = 0.0305 / 2.0
    base_offset = base_half_xy + module_half_xy
    motor_link_offset = 0.038125

    base_faces = (
        {
            "front": {"xyz": f"{base_offset} 0 0", "rpy": "0 0 0"},
            "back": {"xyz": f"{-base_offset} 0 0", "rpy": f"0 0 {math.pi}"},
            "left": {"xyz": f"0 {base_offset} 0", "rpy": f"0 0 {math.pi / 2}"},
            "right": {"xyz": f"0 {-base_offset} 0", "rpy": f"0 0 {-math.pi / 2}"},
        }
        if not robot_line
        else {
            "front": {"xyz": f"{base_offset} 0 0", "rpy": "0 0 0"},
        }
    )

    module_faces = (
        {
            "front": {"xyz": f"{motor_link_offset} 0 0", "rpy": "0 0 0"},
            "left": {"xyz": f"0 {motor_link_offset} 0", "rpy": f"0 0 {math.pi / 2}"},
            "right": {"xyz": f"0 {-motor_link_offset} 0", "rpy": f"0 0 {-math.pi / 2}"},
        }
        if not robot_line
        else {
            "front": {"xyz": f"{motor_link_offset} 0 0", "rpy": "0 0 0"},
        }
    )

    robot = Element(
        "robot",
        {"xmlns:xacro": "http://www.ros.org/wiki/xacro", "name": "modular_robot"},
    )

    SubElement(robot, "xacro:include", {"filename": "base.xacro"})
    SubElement(robot, "xacro:include", {"filename": "module.xacro"})

    SubElement(robot, "xacro:base", {"name": "base"})

    modules = [{"name": "base", "parent": None, "used_faces": set()}]

    for i in range(num_modules):
        # Build list of candidate parents and their available faces
        candidate_parents = []
        for m in modules:
            if m["name"] == "base":
                available_faces = [
                    f for f in base_faces if f not in m["used_faces"]]
            else:
                available_faces = [
                    f for f in module_faces if f not in m["used_faces"]]
            if available_faces:
                candidate_parents.append((m, available_faces))
        if not candidate_parents:
            break

        # Pick a parent and a face
        parent, available_faces = random.choice(candidate_parents)
        face = random.choice(available_faces)
        parent["used_faces"].add(face)

        module_name = f"robot_{robot_idx}_module_{i}"
        modules.append(
            {"name": module_name, "parent": parent["name"], "used_faces": set()})

        # Create joint connecting parent and module
        if parent["name"] == "base":
            xyz = base_faces[face]["xyz"]
            rpy = base_faces[face]["rpy"]
            parent_link = "base_link"
        else:
            xyz = module_faces[face]["xyz"]
            rpy = module_faces[face]["rpy"]
            parent_link = f"{parent['name']}_motor_link"

        joint_name = f"joint_{parent['name']}_{module_name}"
        joint = SubElement(
            robot, "joint", {"name": joint_name, "type": "fixed"})
        SubElement(joint, "parent", {"link": parent_link})
        SubElement(joint, "child", {"link": f"{module_name}_connector_link"})
        SubElement(joint, "origin", {"xyz": xyz, "rpy": rpy})
        SubElement(robot, "xacro:module", {"name": module_name})

    xml_str = minidom.parseString(tostring(robot, encoding="unicode")).toprettyxml(
        indent="  "
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)


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

    simulate(gen_size=args.robots, num_modules=args.modules)
