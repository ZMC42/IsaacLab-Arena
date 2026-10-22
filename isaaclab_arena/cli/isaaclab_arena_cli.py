# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.dataclass_cli import dataclass_from_cli
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.utils.physics_backend import (
    PHYSICS_BACKEND_NAMES,
    PhysicsBackend,
    canonical_physics_backend,
    resolve_physics_backend,
    validate_physics_device,
)


def _physics_backend_from_argparse(args_cli: argparse.Namespace) -> PhysicsBackend | None:
    backend = resolve_physics_backend(getattr(args_cli, "physics_backend", None), getattr(args_cli, "presets", None))
    if backend is not None:
        validate_physics_device(backend, args_cli.device)
    return backend


def requested_physics_backend_from_argparse(args_cli: argparse.Namespace) -> str | None:
    """Return the public name selected by either physics CLI option.

    Args:
        args_cli: Parsed Arena command-line arguments.

    Returns:
        The requested backend name, or None when the environment chooses its default.
    """
    backend = _physics_backend_from_argparse(args_cli)
    return canonical_physics_backend(backend) if backend is not None else None


class _ArenaArgumentParser(argparse.ArgumentParser):
    """Reject incompatible physics selections before starting the simulation app."""

    def parse_known_args(self, args=None, namespace=None):
        args_cli, unknown = super().parse_known_args(args, namespace)
        try:
            _physics_backend_from_argparse(args_cli)
        except ValueError as exc:
            self.error(str(exc))
        return args_cli, unknown


# TODO(cvolk, 2026-07-03): [typed-config-migration] Delete this Namespace-to-config adapter after policy_runner,
# experiment_runner, and the remaining argparse scripts pass ArenaEnvBuilderCfg directly.
def arena_env_builder_cfg_from_argparse(args_cli: argparse.Namespace) -> ArenaEnvBuilderCfg:
    """Translate parsed CLI arguments into the typed builder configuration.

    Args:
        args_cli: Parsed Arena and Isaac Lab command-line arguments.

    Returns:
        The configuration consumed by ``ArenaEnvBuilder``.
    """
    presets = _physics_backend_from_argparse(args_cli)
    cfg = dataclass_from_cli(ArenaEnvBuilderCfg, args_cli)
    cfg.presets = presets
    return cfg


# TODO(cvolk, 2026-07-03): [typed-config-migration] Delete this parser pipeline and its add_* helpers after
# policy_runner, experiment_runner, and the remaining argparse scripts accept typed configs.
def get_isaaclab_arena_cli_parser() -> argparse.ArgumentParser:
    """Get a complete argument parser with both Isaac Lab and IsaacLab Arena arguments."""
    parser = _ArenaArgumentParser(description="IsaacLab Arena CLI parser.")
    AppLauncher.add_app_launcher_args(parser)
    add_isaac_lab_cli_args(parser)
    add_isaaclab_arena_cli_args(parser)
    add_external_environments_cli_args(parser)
    add_env_graph_spec_cli_args(parser)
    return parser


def add_isaac_lab_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add the existing Isaac Lab builder and distributed CLI flags."""

    isaac_lab_group = parser.add_argument_group("Isaac Lab Arguments", "Arguments specific to Isaac Lab framework")

    # TODO(cvolk, 2026-07-06): [typed-config-migration] Delete these manual builder flags after runner scripts
    # receive ArenaEnvBuilderCfg directly. The adapter tests keep their defaults aligned
    # with ArenaEnvBuilderCfg during the transition.
    isaac_lab_group.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    isaac_lab_group.add_argument("--seed", type=int, default=42, help="Optional seed for the random number generator.")
    isaac_lab_group.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    isaac_lab_group.add_argument("--env_spacing", type=float, default=30.0, help="Spacing between environments.")
    isaac_lab_group.add_argument("--mimic", action="store_true", default=False, help="Enable mimic environment.")

    # TODO(cvolk, 2026-07-06): [typed-config-migration] Move --distributed into a typed runner or simulation-app
    # config. It controls AppLauncher and policy_runner process setup, not ArenaEnvBuilder.
    isaac_lab_group.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="Run distributed (one process per GPU). Use with torchrun; AppLauncher uses LOCAL_RANK for device.",
    )


def add_isaaclab_arena_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add Isaac Lab Arena specific command line arguments to the given parser."""
    arena_group = parser.add_argument_group(
        "Isaac Lab Arena Arguments", "Arguments specific to Isaac Lab Arena framework"
    )

    # Arena builds its environment after AppLauncher starts, so this flag both selects
    # optional environment cameras and enables camera rendering during app startup.
    arena_group.add_argument(
        "--enable_cameras",
        action="store_true",
        default=False,
        help="Add the environment's camera sensors and observations.",
    )

    # TODO(cvolk, 2026-07-06): [typed-config-migration] Delete these manual builder flags after runner scripts
    # receive ArenaEnvBuilderCfg directly. The adapter tests keep their defaults aligned
    # with ArenaEnvBuilderCfg during the transition.
    arena_group.add_argument(
        "--no_solve_relations",
        action="store_false",
        dest="solve_relations",
        default=True,
        help="Disable solving spatial relations in the environment.",
    )
    arena_group.add_argument(
        "--placement_layouts",
        dest="placement_layouts_path",
        type=str,
        default=None,
        help="Replay a companion placement JSONL instead of solving; path is relative to the working directory.",
    )
    arena_group.add_argument(
        "--placement_seed",
        type=int,
        default=None,
        help="Seed for solved object placement. Not supported with cached placement layouts.",
    )
    arena_group.add_argument(
        "--physics-backend",
        choices=list(PHYSICS_BACKEND_NAMES),
        default=None,
        help="Physics backend: isaacsim_physx (CPU/CUDA) or newton_mjwarp (CUDA only).",
    )
    arena_group.add_argument(
        "--presets",
        type=PhysicsBackend,
        choices=list(PhysicsBackend),
        default=None,
        help="Arena physics backend preset: 'physx' or 'newton'. When not set, each environment uses its own default.",
    )
    arena_group.add_argument(
        "--resolve_on_reset",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Re-place objects from the pool on each reset (default: True). Use --no-resolve_on_reset to keep the same"
            " layout."
        ),
    )
    arena_group.add_argument(
        "--list_variations",
        action="store_true",
        default=False,
        help="Print Hydra-configurable variations for the selected environment and exit.",
    )


def add_env_graph_spec_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add environment graph spec specific command line arguments to the given parser."""
    env_graph_spec_group = parser.add_argument_group(
        "Environment Graph Spec Arguments", "Arguments specific to environment graph spec"
    )
    env_graph_spec_group.add_argument(
        "--env_spec",
        type=str,
        default=None,
        help=(
            "Path to an environment graph spec YAML. When set, the environment is built from the graph spec instead of"
            " a registered example-environment name; the env-name subcommand then becomes optional. Any override flags"
            " the YAML declares under `cli_override_specs` are added to the parser dynamically."
        ),
    )


def add_external_environments_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add external environments specific command line arguments to the given parser."""
    external_environments_group = parser.add_argument_group(
        "External Environments Arguments", "Arguments specific to external environments"
    )
    external_environments_group.add_argument(
        "--external_environment_class_path",
        type=str,
        default=None,
        help="Name of the external environment to run",
    )
