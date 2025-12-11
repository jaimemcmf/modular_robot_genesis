import argparse
from pathlib import Path
import random
from xml.etree.ElementTree import Element, SubElement, tostring
import math
import genesis as gs
from itertools import chain
import numpy as np
from helpers import build_urdf, build_random_tree
import torch
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

    for i in range(len(robots)):
        (urdf_dir / f"random_robot_{i}.urdf.xacro").unlink(missing_ok=True)
        (urdf_dir / f"random_robot_{i}.urdf").unlink(missing_ok=True)
    
def selection(robot_gen):

    new_gen = sorted(robot_gen, key=lambda r: r.fitness,
                     reverse=True)[:len(robot_gen)//2]
    return new_gen

def mutation(robots):

    p_num_modules = 0.3     
    p_phase = 0.15           
    p_amplitude = 0.15        
    p_frequency = 0.15        


    phase_sigma = 0.2         # radians
    amplitude_scale_sigma = 0.2
    frequency_scale_sigma = 0.2

    for robot in robots:
        if random.random() < p_num_modules:
            change = random.choice([-1, 1])
            new_num = robot.num_modules + change
            if new_num >= 1:
                if change == 1:
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
                    robot.phases = robot.phases[:-1]
                    robot.amplitudes = robot.amplitudes[:-1]
                    robot.frequencies = robot.frequencies[:-1]
                robot.num_modules = new_num

        n = robot.num_modules
        if robot.phases.numel() != n:
            robot.phases = robot.phases[:n]
        if robot.amplitudes.numel() != n:
            robot.amplitudes = robot.amplitudes[:n]
        if robot.frequencies.numel() != n:
            robot.frequencies = robot.frequencies[:n]

        if n > 0:
            mask_phase = torch.rand(n) < p_phase
            noise_phase = torch.randn(n) * phase_sigma
            robot.phases = torch.where(
                mask_phase, robot.phases + noise_phase, robot.phases)

            mask_amp = torch.rand(n) < p_amplitude
            scale_amp = 1 + torch.randn(n) * amplitude_scale_sigma
            mutated_amp = torch.clamp(robot.amplitudes * scale_amp, 0.0, 1.0)
            robot.amplitudes = torch.where(
                mask_amp, mutated_amp, robot.amplitudes)

            mask_freq = torch.rand(n) < p_frequency
            scale_freq = 1 + torch.randn(n) * frequency_scale_sigma
            mutated_freq = torch.clamp(
                robot.frequencies * scale_freq, 0.1, 5.0)
            robot.frequencies = torch.where(
                mask_freq, mutated_freq, robot.frequencies)

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

        child1_modules = random.choice([p1.num_modules, p2.num_modules])
        child1 = Robot(id=-1, num_modules=child1_modules)  # id assigned later
        child1.phases = mix_arrays(p1.phases, p2.phases, child1.num_modules)
        child1.amplitudes = mix_arrays(
            p1.amplitudes, p2.amplitudes, child1.num_modules)
        child1.frequencies = mix_arrays(
            p1.frequencies, p2.frequencies, child1.num_modules)
        child1.fitness = 0.0
        child1.entity = None
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

    for i, c in enumerate(children):
        c.id = i

    return children


def main(num_generations=5, generation_size=8):
    generations = 5
    gen_size = 8

    gs.init(backend=gs.cpu)

    scene = init_genesis()
    robot_gen = initialization(gen_size)
    add_gen_to_scene(scene, robot_gen)
    scene.build()

    print(f"Evaluating generation 0 with {len(robot_gen)} robots...")
    evaluate(scene, robot_gen)
    print("Initial fitnesses:", [round(r.fitness, 4) for r in robot_gen])

    for g in range(1, generations + 1):

        selected = selection(robot_gen)
        new_gen = crossover(selected, gen_size)
        new_gen = mutation(new_gen)

        scene = init_genesis()
        add_gen_to_scene(scene, new_gen)
        scene.build()

        print(f"Evaluating generation {g} with {len(new_gen)} robots...")
        evaluate(scene, new_gen)
        print("Fitnesses:", [round(r.fitness, 4) for r in new_gen])

        robot_gen = new_gen

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolutionary optimization for modular robots")
    parser.add_argument("--generations", "-g", type=int, default=3, help="Number of generations to run")
    parser.add_argument("--gen-size", "-n", type=int, default=4, help="Number of robots per generation")
    args = parser.parse_args()

    main(num_generations=args.generations, generation_size=args.gen_size)