import subprocess
from pathlib import Path
from urdfpy import URDF
import random
import json
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import math

project_root = Path(__file__).resolve().parents[1]
urdf_dir = project_root / "urdf"
xacro_file = urdf_dir / "modular_robot.urdf.xacro"
# xacro_file = urdf_dir / "random_robot.xacro"
urdf_file = urdf_dir / "modular_robot.urdf"


def simulate():
    import genesis as gs  # imported inside the function because it takes long to load
    import numpy as np
    import torch

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
        show_viewer=True,
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            dt=0.01,
            enable_adjacent_collision=True,
            enable_joint_limit=True,
            gravity=np.array([0.0, -9.81, 0.0]),
            constraint_solver=gs.constraint_solver.Newton
        )
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.URDF(
        file=str(urdf_file), pos=(0, 0, 0)))
    scene.build()

    initial_position = robot.get_pos()
    print("Robot initial position:", initial_position)
    jnt_names = ["emerge_plus_joint", "emerge_minus_joint", "emerge_y_joint"]
    dofs_idx = [robot.get_joint(name).dof_idx for name in jnt_names]

    robot.set_dofs_kp(np.array([80.0, 80.0, 80.0]), dofs_idx)
    robot.set_dofs_kv(np.array([2.0, 2.0, 2.0]), dofs_idx)
    robot.set_dofs_force_range(
        np.array([-3.0, -3.0, -3.0]), np.array([3.0, 3.0, 3.0]), dofs_idx)

    # initial position target
    t = 0.0
    dt = 0.01
    frequency = 1.0  # 20 Hz oscillation
    amplitude = 0.5  # radians

    for _ in range(1000):

        t += dt

        target1 = amplitude * np.sin(2*np.pi*frequency*t)
        target2 = amplitude * np.sin(2*np.pi*frequency*t + np.pi/2)
        target_3 = amplitude * np.sin(2*np.pi*frequency*t + np.pi)

        targets = np.array([target1, target2, target_3])

        # Send target positions to both joints
        robot.control_dofs_position(targets, dofs_idx)

        scene.step()

    final_position = robot.get_pos()

    print("Initial positions:", initial_position)
    print("Final positions:", final_position)
    print("Distance:", torch.norm(
        final_position - initial_position).absolute().item())


def check_robot():
    robot = URDF.load(urdf_file)
    print(robot)
    print(len(robot.links), len(robot.joints))


def build_urdf():
    if xacro_file.exists():
        print("[INFO] Expanding Xacro → URDF...")
        try:
            subprocess.run(
                ["xacro", "--inorder", str(xacro_file), "-o", str(urdf_file)],
                check=True,
            )
            print(f"[OK] Generated: {urdf_file}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to generate URDF from Xacro:\n{e}")


def build_random_tree(num_modules: int, output_path: str = "urdf/random_robot.urdf.xacro"):
    """
    Builds a random modular robot Xacro with a flat base as root and N emerge_modules attached.
    Saves the generated XML to `output_path`.
    """

    # Define possible connection faces and their orientations
    flat_half_x = 0.0747 / 2.0
    emerge_half_x = 0.0305 / 2.0
    base_offset = flat_half_x + emerge_half_x
    base_faces = {
        "front":  {"xyz": f"{base_offset} 0 0",  "rpy": "0 0 0"},
        "back":   {"xyz": f"{-base_offset} 0 0", "rpy": f"0 0 {math.pi}"},
        "left":   {"xyz": f"0 {base_offset} 0", "rpy": f"0 0 {math.pi/2}"},
        "right":  {"xyz": f"0 {-base_offset} 0",  "rpy": f"0 0 {-math.pi/2}"},
    }

    robot = Element("robot", {
                    "xmlns:xacro": "http://www.ros.org/wiki/xacro", "name": "modular_robot"})

    SubElement(robot, "xacro:include", {"filename": "base.xacro"})
    SubElement(robot, "xacro:include", {"filename": "module.xacro"})

    SubElement(robot, "xacro:base", {"name": "base"})

    modules = [{"name": "base", "parent": None, "used_faces": set()}]

    for i in range(num_modules):
        # Pick a valid parent (flat base or any emerge module with free faces)
        possible_parents = [m for m in modules if len(
            m["used_faces"]) < (4 if m["name"] == "base" else 3)]
        if not possible_parents:
            print("⚠️ No more free attachment faces — stopping early.")
            break

        parent = random.choice(possible_parents)
        available_faces = [
            f for f in base_faces if f not in parent["used_faces"]]
        if not available_faces:
            continue

        face = random.choice(available_faces)
        parent["used_faces"].add(face)

        module_name = f"module_{i}"
        modules.append(
            {"name": module_name, "parent": parent["name"], "used_faces": set()})

        # Create joint connecting parent and module
        xyz = base_faces[face]["xyz"]
        rpy = base_faces[face]["rpy"]

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
        SubElement(robot, "xacro:module", {
                   "name": module_name})  # instance of module

    xml_str = minidom.parseString(
        tostring(robot, encoding="unicode")).toprettyxml(indent="  ")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

if __name__ == "__main__":
    """ build_urdf()
    simulate() """
    build_random_tree(2)
