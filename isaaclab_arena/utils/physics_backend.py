# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Physics backend identifiers shared across Arena layers."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class PhysicsBackend(str, Enum):
    """Physics backends supported by Arena."""

    PHYSX = "physx"
    NEWTON = "newton"

    def __str__(self) -> str:
        return self.value


def canonical_physics_backend(backend: PhysicsBackend) -> str:
    """Return the full physics implementation name for an Arena preset."""
    return {PhysicsBackend.PHYSX: "isaacsim_physx", PhysicsBackend.NEWTON: "newton_mjwarp"}[backend]


def validate_physics_device(backend: PhysicsBackend, device: str) -> None:
    """Require a CUDA execution device for the Newton MuJoCo Warp backend.

    Args:
        backend: Selected Arena physics backend.
        device: Physics execution device.
    """
    is_cuda = device == "cuda" or (device.startswith("cuda:") and device[5:].isdigit())
    if backend is PhysicsBackend.NEWTON and not is_cuda:
        raise ValueError(f"newton_mjwarp requires a CUDA device, got {device!r}")


def physics_protocol(env: ManagerBasedEnv) -> dict[str, Any]:
    """Describe the actual simulator configuration after environment creation.

    Args:
        env: The running, unwrapped environment.

    Returns:
        JSON-serializable physics, camera, and package configuration.
    """
    from importlib.metadata import version

    from isaaclab.sensors import CameraCfg
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_physx.physics import PhysxCfg

    physics = env.cfg.sim.physics
    backend = f"{type(physics).__module__}:{type(physics).__qualname__}"
    if isinstance(physics, PhysxCfg):
        backend = "isaacsim_physx"
    elif (
        isinstance(physics, NewtonCfg)
        and isinstance(physics.solver_cfg, MJWarpSolverCfg)
        and not physics.solver_cfg.use_mujoco_cpu
    ):
        backend = "newton_mjwarp"
    manager = env.sim.physics_manager
    cameras = {}
    for name, sensor in env.scene.sensors.items():
        if isinstance(sensor.cfg, CameraCfg):
            cameras[name] = {
                "class": type(sensor).__name__,
                "height": sensor.cfg.height,
                "width": sensor.cfg.width,
                "renderer": sensor.cfg.renderer_cfg.to_dict(),
            }
    return {
        "physics_backend": backend,
        "physics_device": str(env.sim.device),
        "physics": type(physics).__name__,
        "physics_manager": f"{manager.__module__}:{manager.__qualname__}",
        "physics_parameters": physics.to_dict(),
        "physics_dt": env.cfg.sim.dt,
        "step_dt": env.step_dt,
        "decimation": env.cfg.decimation,
        "cameras": cameras,
        "packages": {
            name: version(name)
            for name in (
                "torch",
                "isaacsim",
                "isaaclab",
                "isaaclab_arena",
                "newton",
                "mujoco-warp",
                "warp-lang",
            )
        },
    }
