import subprocess
from urdfpy import URDF
import random
import math
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import genesis as gs
import numpy as np
import time
import psutil
import os
import torch

def init_genesis(viewer: bool = False):
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
        show_viewer=viewer,
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            dt=0.01,
            enable_joint_limit=True,
            gravity=np.array([0.0, -9.81, 0.0]),
            constraint_solver=gs.constraint_solver.Newton,
        ),
    )
    scene.add_entity(gs.morphs.Plane())

    return scene

def check_robot(path):
    robot = URDF.load(path)
    print(robot)
    print(len(robot.links), len(robot.joints))


def build_urdf(path):
    try:
        subprocess.run(
            ["xacro", str(path), "-o", str(path.with_suffix(""))],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to generate URDF from Xacro:\n{e}")

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
        
def get_cpu_mem_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

""" def get_gpu_mem_mb():
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2

    # torch.cuda.reset_peak_memory_stats() """
    
import subprocess
import os

def get_gpu_mem_mb():
    import subprocess
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
        # If multiple GPUs, take the first one
        return float(out.strip().split("\n")[0])
    except Exception:
        return 0.0

