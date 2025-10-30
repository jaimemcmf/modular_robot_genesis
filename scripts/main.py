import subprocess
from pathlib import Path
import time
from urdfpy import URDF

project_root = Path(__file__).resolve().parents[1]
urdf_dir = project_root / "urdf"
xacro_file = urdf_dir / "modular_robot.urdf.xacro"
urdf_file = urdf_dir / "modular_robot.urdf"

def simulate():
    import genesis as gs #import is inside because it is slow to load
    
    gs.init(backend=gs.cpu)

    scene = gs.Scene(show_viewer=True)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.URDF(file=str(urdf_file))
    
    )

    scene.build()

    print("[INFO] Starting simulation...")
    for _ in range(500):
        scene.step()
        time.sleep(1.0 / 60.0)
    print("[INFO] Simulation finished.")
    
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
    