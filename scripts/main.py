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
from dataclasses import dataclass

project_root = Path(__file__).resolve().parents[1]
urdf_dir = project_root / "urdf"


@dataclass
class Robot:

    id: int
    num_modules: int
    entity: any
    fitness: float

    def __init__(self, id, num_modules):
        self.id = id
        self.num_modules = num_modules
        self.fitness = 0.0

    def __repr__(self):
        return f"Robot(id={self.id}, num_modules={self.num_modules})"


def initialization(gen_size: int):
    # initial ranges
    num_modules_range = (1, 5)
    phase_range = (0.0, 2 * math.pi)
    amplitude_range = (0.4, 0.6)
    frequency_range = (0.8, 1.2)

    robots = []

    for i in range(gen_size):
        num_modules = random.randint(*num_modules_range)
        phases = torch.FloatTensor(num_modules).uniform_(*phase_range)
        amplitudes = torch.FloatTensor(num_modules).uniform_(*amplitude_range)
        frequencies = torch.FloatTensor(num_modules).uniform_(*frequency_range)

        robot = Robot(id=i, num_modules=num_modules)
        robot.phases = phases
        robot.amplitudes = amplitudes
        robot.frequencies = frequencies
        robots.append(robot)

    return robots


def init_genesis():
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

    return scene


def add_gen_to_scene(scene, robot_gen):
    for i, robot in enumerate(robot_gen):
        xacro_file = urdf_dir / f"random_robot_{robot.id}.urdf.xacro"
        build_random_tree(
            num_modules=robot.num_modules,
            robot_idx=i,
            output_path=str(xacro_file),
            robot_line=False
        )
        build_urdf(xacro_file)
        possible_locations = [(0.5 * (i - 2), 0, 0), (0, 0.5 * (i - 2), 0)]
        entity = scene.add_entity(
            gs.morphs.URDF(
                file=str(xacro_file.with_suffix("")),
                pos=random.choice(possible_locations),
            )
        )
        robot.entity = entity


def write_fitnesses(gen_idx: int, robots, out_path: Path):
    """
    Append fitnesses of a generation to a CSV file.
    Format per line: gen_idx,robot_id,modules,fitness
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for r in robots:
            f.write(f"{gen_idx},{r.id},{r.num_modules},{r.fitness:.6f}\n")


def main():
    # Config
    generations = 5
    gen_size = 8

    # Init Genesis runtime once
    gs.init(backend=gs.cpu)

    # Initialize generation
    scene = init_genesis()
    robot_gen = initialization(gen_size)
    add_gen_to_scene(scene, robot_gen)
    scene.build()

    # Evaluate initial generation
    print(f"Evaluating generation 0 with {len(robot_gen)} robots...")
    evaluate(scene, robot_gen)
    print("Initial fitnesses:", [round(r.fitness, 4) for r in robot_gen])

    # Evolution loop
    for g in range(1, generations + 1):
        # Select, crossover, mutate
        selected = selection(robot_gen)
        new_gen = crossover(selected, gen_size)
        new_gen = mutation(new_gen)

        # Build a fresh scene for the new generation
        scene = init_genesis()
        add_gen_to_scene(scene, new_gen)
        scene.build()

        print(f"Evaluating generation {g} with {len(new_gen)} robots...")
        evaluate(scene, new_gen)
        print("Fitnesses:", [round(r.fitness, 4) for r in new_gen])

        # Log fitnesses for this generation
        write_fitnesses(gen_idx=g, robots=new_gen, out_path=log_path)

        # Prepare for next iteration
        robot_gen = new_gen

    # Report best robot from final generation
    best = max(robot_gen, key=lambda r: r.fitness)
    print(
        f"Best robot: id={best.id}, modules={best.num_modules}, fitness={best.fitness:.4f}")


def evaluate(scene, robots):
    initial_positions = [robot.entity.get_pos() for robot in robots]

    # Prepare per-robot control params and dof indices
    control_params = []  # list of dicts per robot
    for robot in robots:
        dofs_idx = list(
            chain.from_iterable(
                [
                    joint.dofs_idx_local
                    for joint in robot.entity.joints  # might be robot.joints
                    if "module" in joint.name
                ]
            )
        )
        robot.entity.dofs_idx = dofs_idx
        n = len(dofs_idx)
        if n == 0:
            control_params.append(
                {"phases": np.array([]), "amplitudes": np.array(
                    []), "frequencies": np.array([])}
            )
            continue

        robot.entity.set_dofs_kp(np.full(n, 80.0), dofs_idx)
        robot.entity.set_dofs_kv(np.full(n, 2.0), dofs_idx)
        robot.entity.set_dofs_force_range(
            np.full(n, -3.0), np.full(n, 3.0), dofs_idx)

        phases = robot.phases.numpy()
        amplitudes = robot.amplitudes.numpy()
        frequencies = robot.frequencies.numpy()

        control_params.append(
            {"phases": phases, "amplitudes": amplitudes, "frequencies": frequencies}
        )

    t = 0.0
    dt = 0.01
    for _ in range(1000):
        t += dt
        for robot, params in zip(robots, control_params):
            if len(robot.entity.dofs_idx) == 0:
                continue
            targets = params["amplitudes"] * np.sin(
                2 * np.pi * params["frequencies"] * t + params["phases"]
            )
            robot.entity.control_dofs_position(targets, robot.entity.dofs_idx)
        scene.step()

    final_positions = [robot.entity.get_pos() for robot in robots]

    distances = torch.stack(final_positions) - torch.stack(initial_positions)
    distances = torch.norm(distances, dim=1)

    for robot, dist in zip(robots, distances):
        robot.fitness = float(dist.item())

    print("Distances:", [r.fitness for r in robots])

    # Clean up generated files; ignore if already removed
    for i in range(len(robots)):
        (urdf_dir / f"random_robot_{i}.urdf.xacro").unlink(missing_ok=True)
        (urdf_dir / f"random_robot_{i}.urdf").unlink(missing_ok=True)


def selection(robot_gen):
    # elitism: select top 50%
    new_gen = sorted(robot_gen, key=lambda r: r.fitness,
                     reverse=True)[:len(robot_gen)//2]
    return new_gen


def mutation(robots):
    # probabilities
    p_num_modules = 0.3       # chance to mutate the module count
    p_phase = 0.15            # per-joint phase mutation probability
    p_amplitude = 0.15        # per-joint amplitude mutation probability
    p_frequency = 0.15        # per-joint frequency mutation probability

    # mutation scales
    phase_sigma = 0.2         # radians
    amplitude_scale_sigma = 0.2
    frequency_scale_sigma = 0.2

    for robot in robots:
        # Mutate morphology: num_modules (+/- 1) with probability
        if random.random() < p_num_modules:
            change = random.choice([-1, 1])
            new_num = robot.num_modules + change
            if new_num >= 1:
                if change == 1:
                    # Grow by 1: append values based on last element with noise
                    def _append_noise(vec: torch.Tensor):
                        base = vec[-1] if vec.numel() > 0 else torch.tensor(0.5,
                                                                            dtype=torch.float32)
                        noise = torch.randn(1) * amplitude_scale_sigma
                        return torch.cat([vec, base.view(1) * (1 + noise)])
                    robot.phases = torch.cat([robot.phases,
                                              robot.phases[-1:].clone() + torch.randn(1) * phase_sigma])
                    robot.amplitudes = _append_noise(robot.amplitudes)
                    robot.frequencies = torch.cat([robot.frequencies,
                                                   robot.frequencies[-1:].clone() * (1 + torch.randn(1) * frequency_scale_sigma)])
                else:
                    # Shrink by 1: drop last element
                    robot.phases = robot.phases[:-1]
                    robot.amplitudes = robot.amplitudes[:-1]
                    robot.frequencies = robot.frequencies[:-1]
                robot.num_modules = new_num

        # Per-element mutations for control tensors
        n = robot.num_modules
        # Ensure tensors match current num_modules length
        if robot.phases.numel() != n:
            robot.phases = robot.phases[:n]
        if robot.amplitudes.numel() != n:
            robot.amplitudes = robot.amplitudes[:n]
        if robot.frequencies.numel() != n:
            robot.frequencies = robot.frequencies[:n]

        # Phase: add Gaussian noise per element with probability
        if n > 0:
            mask_phase = torch.rand(n) < p_phase
            noise_phase = torch.randn(n) * phase_sigma
            robot.phases = torch.where(
                mask_phase, robot.phases + noise_phase, robot.phases)

            # Amplitude: multiplicative noise, keep within [0.0, 1.0]
            mask_amp = torch.rand(n) < p_amplitude
            scale_amp = 1 + torch.randn(n) * amplitude_scale_sigma
            mutated_amp = torch.clamp(robot.amplitudes * scale_amp, 0.0, 1.0)
            robot.amplitudes = torch.where(
                mask_amp, mutated_amp, robot.amplitudes)

            # Frequency: multiplicative noise, keep within [0.1, 5.0]
            mask_freq = torch.rand(n) < p_frequency
            scale_freq = 1 + torch.randn(n) * frequency_scale_sigma
            mutated_freq = torch.clamp(
                robot.frequencies * scale_freq, 0.1, 5.0)
            robot.frequencies = torch.where(
                mask_freq, mutated_freq, robot.frequencies)

        # Reset fitness after mutation
        robot.fitness = 0.0

    return robots


def crossover(selected_robots, gen_size: int):

    if len(selected_robots) != gen_size // 2:
        raise ValueError("Number of selected robots must be half of gen_size")

    pairs = [(tuple(random.sample(selected_robots, 2)))
             for _ in range(gen_size // 2)]
    children = []

    def mix_arrays(arr1, arr2, num_modules):
        arr1 = np.asarray(arr1)
        arr2 = np.asarray(arr2)

        if len(arr1) < num_modules:
            arr1 = np.pad(arr1, (0, num_modules - len(arr1)),
                          'constant', constant_values=0)
        if len(arr2) < num_modules:
            arr2 = np.pad(arr2, (0, num_modules - len(arr2)),
                          'constant', constant_values=0)

        mask = np.random.rand(num_modules) < 0.5
        mixed = np.where(mask, arr1[:num_modules], arr2[:num_modules])
        return torch.as_tensor(mixed, dtype=torch.float32)

    for p1, p2 in pairs:
        # Create child1 without copying the Taichi entity
        child1_modules = random.choice([p1.num_modules, p2.num_modules])
        child1 = Robot(id=-1, num_modules=child1_modules)  # id assigned later
        child1.phases = mix_arrays(p1.phases, p2.phases, child1.num_modules)
        child1.amplitudes = mix_arrays(
            p1.amplitudes, p2.amplitudes, child1.num_modules)
        child1.frequencies = mix_arrays(
            p1.frequencies, p2.frequencies, child1.num_modules)
        child1.fitness = 0.0
        child1.entity = None  # will be set when added to the scene
        children.append(child1)

        # Create child2 similarly
        child2_modules = random.choice([p1.num_modules, p2.num_modules])
        child2 = Robot(id=-1, num_modules=child2_modules)
        child2.phases = mix_arrays(p1.phases, p2.phases, child2.num_modules)
        child2.amplitudes = mix_arrays(
            p1.amplitudes, p2.amplitudes, child2.num_modules)
        child2.frequencies = mix_arrays(
            p1.frequencies, p2.frequencies, child2.num_modules)
        child2.fitness = 0.0
        child2.entity = None
        children.append(child2)

    # Assign sequential ids to children for file naming/placement
    for i, c in enumerate(children):
        c.id = i

    return children

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

    # simulate(gen_size=args.robots, num_modules=args.modules)
    main()
