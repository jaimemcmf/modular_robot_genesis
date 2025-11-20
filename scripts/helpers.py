import subprocess
from urdfpy import URDF

def check_robot(path):
    robot = URDF.load(path)
    print(robot)
    print(len(robot.links), len(robot.joints))


def build_urdf(path):
    try:
        subprocess.run(
            ["xacro", "--inorder", str(path), "-o", str(path.with_suffix(''))],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to generate URDF from Xacro:\n{e}")