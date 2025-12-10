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

    for idx, robot in enumerate(robots):
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
        robot.set_dofs_kp(np.full(num_modules, 80.0), dofs_idx)
        robot.set_dofs_kv(np.full(num_modules, 2.0), dofs_idx)
        robot.set_dofs_force_range(
            np.full(num_modules, -3.0), np.full(num_modules, 3.0), dofs_idx
        )

        # initial position target
        t = 0.0
        dt = 0.01
        phases = np.random.uniform(0, 2 * np.pi, num_modules)
        amplitudes = np.random.uniform(0.4, 0.6, num_modules)  # around 0.5
        frequencies = np.random.uniform(0.8, 1.2, num_modules)  # around 1.0 Hz

    for _ in range(1000):
        t += dt
        for robot in robots:
            targets = amplitudes * np.sin(2 * np.pi * frequencies * t + phases)
            robot.control_dofs_position(targets, robot.dofs_idx)

        scene.step()

    final_positions = [robot.get_pos() for robot in robots]

    distances = torch.stack(final_positions) - torch.stack(initial_positions)


    distances = torch.norm(distances, dim=1)   # GPU parallel
    print("Distances:", distances.cpu().tolist())

    for i in range(len(robots)):
        Path.unlink(urdf_dir / f"random_robot_{i}.urdf.xacro")
        Path.unlink(urdf_dir / f"random_robot_{i}.urdf")


def create_robots(scene, gen_size: int, robot_size: int):
    robots = []
    for i in range(gen_size):
        xacro_file = urdf_dir / f"random_robot_{i}.urdf.xacro"
        build_random_tree(
            num_modules=robot_size, robot_idx=i, output_path=str(xacro_file)
        )
        build_urdf(xacro_file)
        possible_locations = [(0.5 * (i - 2), 0, 0), (0, 0.5 * (i - 2), 0)]
        robot = scene.add_entity(
            gs.morphs.URDF(
                file=str(xacro_file.with_suffix("")),
                pos=random.choices(possible_locations)[0],
            )
        )
        robots.append(robot)
    return robots


def build_random_tree(
    num_modules: int, robot_idx="x", output_path: str = "urdf/random_robot.urdf.xacro"
):
    base_half_xy = 0.0747 / 2.0
    module_half_xy = 0.0305 / 2.0
    base_offset = base_half_xy + module_half_xy
    motor_link_offset = 0.038125

    base_faces = {
        "front": {"xyz": f"{base_offset} 0 0", "rpy": "0 0 0"},
        "back": {"xyz": f"{-base_offset} 0 0", "rpy": f"0 0 {math.pi}"},
        "left": {"xyz": f"0 {base_offset} 0", "rpy": f"0 0 {math.pi / 2}"},
        "right": {"xyz": f"0 {-base_offset} 0", "rpy": f"0 0 {-math.pi / 2}"},
    }

    module_faces = {
        "front": {"xyz": f"{motor_link_offset} 0 0", "rpy": "0 0 0"},
        "left": {"xyz": f"0 {motor_link_offset} 0", "rpy": f"0 0 {math.pi / 2}"},
        "right": {"xyz": f"0 {-motor_link_offset} 0", "rpy": f"0 0 {-math.pi / 2}"},
    }

    robot = Element(
        "robot",
        {"xmlns:xacro": "http://www.ros.org/wiki/xacro", "name": "modular_robot"},
    )

    SubElement(robot, "xacro:include", {"filename": "base.xacro"})
    SubElement(robot, "xacro:include", {"filename": "module.xacro"})

    SubElement(robot, "xacro:base", {"name": "base"})

    modules = [{"name": "base", "parent": None, "used_faces": set()}]

    for i in range(num_modules):
        # Pick a valid parent (flat base or any emerge module with free faces)
        possible_parents = [
            m
            for m in modules
            if len(m["used_faces"]) < (4 if m["name"] == "base" else 3)
        ]
        if not possible_parents:
            break

        parent = random.choice(possible_parents)
        if parent["name"] == "base":
            available_faces = [
                f for f in base_faces if f not in parent["used_faces"]]
        else:
            available_faces = [
                f for f in module_faces if f not in parent["used_faces"]]
        if not available_faces:
            continue

        face = random.choice(available_faces)
        parent["used_faces"].add(face)

        module_name = f"robot_{robot_idx}_module_{i}"
        modules.append(
            {"name": module_name,
                "parent": parent["name"], "used_faces": set()}
        )

        # Create joint connecting parent and module
        if parent["name"] == "base":
            xyz = base_faces[face]["xyz"]
            rpy = base_faces[face]["rpy"]
        else:
            xyz = module_faces[face]["xyz"]
            rpy = module_faces[face]["rpy"]

        joint_name = f"joint_{parent['name']}_{module_name}"
        joint = SubElement(
            robot, "joint", {"name": joint_name, "type": "fixed"})

        if parent["name"] == "base":
            SubElement(joint, "parent", {"link": "base_link"})
        else:
            SubElement(joint, "parent", {
                       "link": f"{parent['name']}_motor_link"})
        # child link, always the connector link
        SubElement(joint, "child", {"link": f"{module_name}_connector_link"})
        # position and orientation
        SubElement(joint, "origin", {"xyz": xyz, "rpy": rpy})
        # instance of module
        SubElement(robot, "xacro:module", {"name": module_name})

    xml_str = minidom.parseString(tostring(robot, encoding="unicode")).toprettyxml(
        indent="  "
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genesis Modular Robot Simulation")

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
