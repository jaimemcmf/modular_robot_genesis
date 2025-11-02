import subprocess
from pathlib import Path
import time
import numpy as np
from urdfpy import URDF

project_root = Path(__file__).resolve().parents[1]
urdf_dir = project_root / "urdf"
xacro_file = urdf_dir / "modular_robot.urdf.xacro"
urdf_file = urdf_dir / "modular_robot.urdf"


def simulate():
    import genesis as gs
    import numpy as np

    gs.init(backend=gs.cpu)

    scene = gs.Scene(show_viewer=True)
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.URDF(file=str(urdf_file)))
    scene.build()

    jnt_names = ["emerge_plus_joint", "emerge_minus_joint"]
    dofs_idx = [robot.get_joint(name).dof_idx_local for name in jnt_names]

    robot.set_dofs_kp(np.array([80.0, 80.0]), dofs_idx)
    robot.set_dofs_kv(np.array([8.0, 8.0]), dofs_idx)
    robot.set_dofs_force_range(np.array([-50.0, -50.0]), np.array([50.0, 50.0]), dofs_idx)

    # initial position target
    t = 0.0
    dt = 0.01
    frequency = 5.0  # 5 Hz oscillation
    amplitude = 0.5  # radians

    for step in range(1000):
        t += dt
        # Both joints have the same sinusoidal target
        target_angle = amplitude * np.sin(2 * np.pi * frequency * t)
        targets = np.array([target_angle, target_angle])  # Second joint remains at 0.0

        # Send target positions to both joints
        robot.control_dofs_position(targets, dofs_idx)

        # Advance simulation by one step
        scene.step()


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


if __name__ == "__main__":
    build_urdf()
    simulate()
