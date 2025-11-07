import subprocess
from pathlib import Path
from urdfpy import URDF

project_root = Path(__file__).resolve().parents[1]
urdf_dir = project_root / "urdf"
xacro_file = urdf_dir / "modular_robot.urdf.xacro"
urdf_file = urdf_dir / "modular_robot.urdf"


def simulate():
    import genesis as gs # imported inside the function because it takes long to load
    import numpy as np

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        viewer_options = gs.options.ViewerOptions(
            camera_pos    = (0, -3.5, 2.5),
            camera_lookat = (0.0, 0.0, 0.5),
            camera_fov    = 30,
            res           = (960, 640),
            max_FPS       = 60,
        ),
        sim_options = gs.options.SimOptions(
            dt = 0.01,
        ),
        show_viewer = True,
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            dt = 0.01,
            enable_adjacent_collision = True,
            enable_joint_limit = True,
            gravity = np.array([0.0, -9.81, 0.0]),
            constraint_solver = gs.constraint_solver.Newton   
        )
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.URDF(file=str(urdf_file), pos=(0, 0, 1.5)))
    scene.build()

    jnt_names = ["emerge_plus_joint", "emerge_minus_joint"]
    dofs_idx = [robot.get_joint(name).dof_idx for name in jnt_names]

    robot.set_dofs_kp(np.array([80.0, 80.0]), dofs_idx)
    robot.set_dofs_kv(np.array([2.0, 2.0]), dofs_idx)
    robot.set_dofs_force_range(np.array([-3.0, -3.0]), np.array([3.0, 3.0]), dofs_idx)


    # initial position target
    t = 0.0
    dt = 0.01
    frequency = 1.0  # 20 Hz oscillation
    amplitude = 0.5  # radians

    for step in range(1000):
        t += dt
        # Both joints have the same sinusoidal target
        target1 = amplitude * np.sin(2*np.pi*frequency*t)
        target2 = amplitude * np.sin(2*np.pi*frequency*t + np.pi/2)
        targets = np.array([target1, target2])

        # Second joint remains at 0.0
        #targets = np.array([target_angle, target_angle])

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
