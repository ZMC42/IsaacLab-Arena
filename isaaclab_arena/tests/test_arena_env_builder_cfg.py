# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the typed Arena environment-builder configuration boundary."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from isaaclab_arena.cli.isaaclab_arena_cli import (
    arena_env_builder_cfg_from_argparse,
    get_isaaclab_arena_cli_parser,
    requested_physics_backend_from_argparse,
)
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.evaluation.policy_runner_cli import add_policy_runner_arguments
from isaaclab_arena.utils.physics_backend import (
    PhysicsBackend,
    canonical_physics_backend,
    physics_backend_from_name,
    physics_protocol,
    resolve_physics_backend,
    validate_physics_device,
)


def test_runtime_protocol_preserves_other_physics_configs():
    """Recording a generic runner must preserve configurations beyond the project selectors."""
    from isaaclab.physics import PhysicsCfg

    physics = PhysicsCfg(class_type=object)
    env = SimpleNamespace(
        cfg=SimpleNamespace(sim=SimpleNamespace(physics=physics, dt=0.005), decimation=4),
        sim=SimpleNamespace(physics_manager=object, device="cuda:0"),
        scene=SimpleNamespace(sensors={}),
        step_dt=0.02,
    )
    protocol = physics_protocol(env)
    assert protocol["physics_backend"] == "isaaclab.physics.physics_manager_cfg:PhysicsCfg"
    assert protocol["physics_parameters"]["class_type"] == "builtins:object"
    json.dumps(protocol)


# TODO(cvolk, 2026-07-03): [typed-config-migration] Delete the argparse adapter tests below with
# arena_env_builder_cfg_from_argparse once runners pass ArenaEnvBuilderCfg directly.
def test_cli_defaults_match_builder_configuration():
    """Keep the manual CLI defaults aligned with the typed builder configuration."""
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])

    assert not hasattr(args_cli, "headless")
    assert args_cli.visualizer is None
    assert arena_env_builder_cfg_from_argparse(args_cli) == ArenaEnvBuilderCfg()


def test_argparse_adapter_maps_builder_configuration():
    """Translate only builder-owned command-line values into the typed config."""
    parser = get_isaaclab_arena_cli_parser()
    add_policy_runner_arguments(parser)
    args_cli = parser.parse_args([
        "--policy_type",
        "zero_action",
        "--num_envs",
        "3",
        "--env_spacing",
        "2.5",
        "--seed",
        "7",
        "--no_solve_relations",
        "--placement_seed",
        "11",
        "--no-resolve_on_reset",
        "--disable_fabric",
        "--mimic",
        "--presets",
        "newton",
        "--device",
        "cuda:0",
        "--language_instruction",
        "pick up the cube",
    ])

    cfg = arena_env_builder_cfg_from_argparse(args_cli)

    assert cfg == ArenaEnvBuilderCfg(
        num_envs=3,
        env_spacing=2.5,
        seed=7,
        solve_relations=False,
        placement_seed=11,
        resolve_on_reset=False,
        disable_fabric=True,
        mimic=True,
        presets=PhysicsBackend.NEWTON,
        device="cuda:0",
        language_instruction="pick up the cube",
    )
    assert cfg.presets is PhysicsBackend.NEWTON


def test_builder_configuration_requires_positive_num_envs():
    """Reject configurations that cannot build any environment instances."""
    with pytest.raises(AssertionError, match="num_envs must be greater than zero"):
        ArenaEnvBuilderCfg(num_envs=0)


def test_builder_configuration_rejects_unknown_physics_backend():
    """Reject arbitrary strings at the typed builder boundary."""
    with pytest.raises(ValueError, match="unknown_backend"):
        ArenaEnvBuilderCfg(presets="unknown_backend")


def test_builder_configuration_normalizes_physics_backend_strings():
    """Normalize legacy configuration input to the typed backend enum."""
    cfg = ArenaEnvBuilderCfg(presets="newton")

    assert cfg.presets is PhysicsBackend.NEWTON


@pytest.mark.parametrize(
    ("name", "backend"),
    [("isaacsim_physx", PhysicsBackend.PHYSX), ("newton_mjwarp", PhysicsBackend.NEWTON)],
)
def test_public_backend_names_map_to_arena_presets(name, backend):
    """Use the same backend identity across public CLI, presets, and run records."""
    assert physics_backend_from_name(name) is backend
    assert canonical_physics_backend(backend) == name
    assert resolve_physics_backend(name, backend) is backend
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--physics-backend", name, "--presets", backend.value])
    assert arena_env_builder_cfg_from_argparse(args_cli).presets is backend


def test_unspecified_backend_preserves_environment_default():
    """Let environments select their own default when neither selector is given."""
    assert resolve_physics_backend() is None


@pytest.mark.parametrize(
    ("arguments", "requested"),
    [
        ([], None),
        (["--presets", "physx"], "isaacsim_physx"),
        (["--presets", "newton"], "newton_mjwarp"),
        (["--physics-backend", "isaacsim_physx"], "isaacsim_physx"),
        (["--physics-backend", "newton_mjwarp", "--presets", "newton"], "newton_mjwarp"),
    ],
)
def test_requested_backend_records_public_and_legacy_selectors(arguments, requested):
    args_cli = get_isaaclab_arena_cli_parser().parse_args(arguments)

    assert requested_physics_backend_from_argparse(args_cli) == requested


def test_backend_validation_keeps_environment_subparser_compatible():
    """Validate the complete namespace after a task subparser consumes its own arguments."""
    parser = get_isaaclab_arena_cli_parser()
    task_parser = parser.add_subparsers(dest="example_environment").add_parser("example")
    task_parser.add_argument("--task-option")

    args_cli = parser.parse_args(["--physics-backend", "newton_mjwarp", "example", "--task-option", "value"])

    assert args_cli.task_option == "value"
    assert arena_env_builder_cfg_from_argparse(args_cli).presets is PhysicsBackend.NEWTON


def test_unknown_public_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown physics backend"):
        physics_backend_from_name("unknown_backend")


def test_conflicting_backend_selectors_are_rejected():
    """Keep legacy preset input from silently overriding the public selection."""
    with pytest.raises(ValueError, match="conflicts"):
        resolve_physics_backend("newton_mjwarp", PhysicsBackend.PHYSX)


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:12"])
def test_newton_accepts_cuda_devices(device):
    validate_physics_device(PhysicsBackend.NEWTON, device)


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:", "cuda:-1", "cuda:invalid"])
def test_newton_rejects_non_cuda_devices(device):
    with pytest.raises(ValueError, match="requires a CUDA device"):
        validate_physics_device(PhysicsBackend.NEWTON, device)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_physx_keeps_cpu_and_cuda_support(device):
    validate_physics_device(PhysicsBackend.PHYSX, device)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--physics-backend", "newton_mjwarp", "--device", "cpu"], "requires a CUDA device"),
        (["--presets", "newton", "--device", "cpu"], "requires a CUDA device"),
        (["--physics-backend", "newton_mjwarp", "--presets", "physx"], "conflicts"),
        (["--presets", "physx", "--physics-backend", "newton_mjwarp"], "conflicts"),
    ],
)
def test_invalid_backend_cli_fails_before_simulation_start(arguments, message):
    """Reject bad selections in a child process without starting Isaac Sim."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser; "
                "get_isaaclab_arena_cli_parser().parse_known_args(sys.argv[1:])"
            ),
            *arguments,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    assert message in result.stderr
