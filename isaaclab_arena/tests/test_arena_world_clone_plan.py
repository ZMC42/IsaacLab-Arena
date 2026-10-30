# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise the geometry lookup against real clone plans without starting a simulator."""

import ast
import numpy as np
from pathlib import Path
from types import SimpleNamespace

import pytest
from isaaclab import cloner


@pytest.fixture
def geometry_lookup():
    """Load the production lookup without importing its USD/simulation dependencies."""
    path = Path(__file__).parents[1] / "environments/arena_world_scene_access.py"
    source = ast.parse(path.read_text())
    lookup_names = {"_get_clone_source_environment_ids", "get_representative_geometry_prim_groups"}
    functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in lookup_names]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
        type_ignores=[],
    )
    scope = {"cloner": cloner}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope["get_representative_geometry_prim_groups"]


def _scene(plan, path, source_paths):
    def get_prim(prim_path):
        return SimpleNamespace(GetPath=lambda: prim_path, IsValid=lambda: prim_path in source_paths)

    return SimpleNamespace(
        clone_plan=plan,
        num_envs=len(plan.topology.world_prototype_layout),
        cfg=SimpleNamespace(object=SimpleNamespace(prim_path=path)),
        env_regex_ns=plan.env_template.format("[^/]+"),
        stage=SimpleNamespace(GetPrimAtPath=get_prim),
    )


@pytest.mark.parametrize("explicit_sources", [False, True])
@pytest.mark.parametrize("env_template", ["/World/envs/env_{}", "/World/scenes/scene_{}"])
def test_variant_sources_cover_every_environment(geometry_lookup, explicit_sources, env_template):
    """Variant identity follows topology, including nonzero first env and explicit sources."""
    declared = env_template.format("[^/]+") + "/Object"
    assets = [
        SimpleNamespace(
            prim_path=declared,
            spawn=SimpleNamespace(spawn_path=f"/World/Prototypes/variant_{i}" if explicit_sources else None),
        )
        for i in range(3)
    ]
    # The third variant is unused. The active variants first occur in envs 1 and 0.
    layout = np.array([1, 0, 1, 0, 1, 0], dtype=np.int32)
    plan = cloner.make_clone_plan(
        assets,
        [(0,), (1,), (2,)],
        6,
        env_template=env_template,
        clone_strategy=lambda weights, count: layout,
    )
    roots = (
        ("/World/Prototypes/variant_0", "/World/Prototypes/variant_1")
        if explicit_sources
        else (env_template.format(1) + "/Object", env_template.format(0) + "/Object")
    )
    source_paths = [root + "/Geometry" for root in roots]
    scene = _scene(plan, "{ENV_REGEX_NS}/Object/Geometry", source_paths)
    groups = geometry_lookup(scene, "object")

    assert [(prim.GetPath(), ids) for prim, ids in groups] == [
        (source_paths[0], (1, 3, 5)),
        (source_paths[1], (0, 2, 4)),
    ]
    assert sorted(env_id for _, ids in groups for env_id in ids) == list(range(6))


def test_nearest_populated_owner_and_repeated_memberships(geometry_lookup):
    """Nested assets own their geometry; a repeated sibling is a distinct placement."""
    assets = [
        SimpleNamespace(prim_path="/World/envs/env_[^/]+/Object"),
        SimpleNamespace(prim_path="/World/envs/env_[^/]+/Object/Part"),
        SimpleNamespace(prim_path="/World/envs/env_[^/]+/Object/Part/Geometry"),
    ]
    plan = cloner.make_clone_plan(assets, [(0, 1, 1), (2,)], 4, weights=[1, 0])
    source_path = "/World/envs/env_0/Object/Part/Geometry"
    scene = _scene(plan, "{ENV_REGEX_NS}/Object/Part/Geometry", [source_path])
    groups = geometry_lookup(scene, "object")
    assert [(prim.GetPath(), ids) for prim, ids in groups] == [(source_path, (0, 1, 2, 3))]
    # The second occurrence receives the distinct Part_1 template but retains its source.
    scene.cfg.object.prim_path = "{ENV_REGEX_NS}/Object/Part_1/Geometry"
    groups = geometry_lookup(scene, "object")
    assert [(prim.GetPath(), ids) for prim, ids in groups] == [(source_path, (0, 1, 2, 3))]


def test_one_variant_in_multiple_world_compositions(geometry_lookup):
    """One source can populate several world prototypes with different other assets."""
    assets = [
        SimpleNamespace(prim_path="/World/envs/env_[^/]+/Object"),
        SimpleNamespace(prim_path="/World/envs/env_[^/]+/Object"),
        SimpleNamespace(prim_path="/World/envs/env_[^/]+/Other"),
        SimpleNamespace(prim_path="/World/envs/env_[^/]+/Other"),
    ]
    layout = np.array([3, 0, 1, 2, 0, 3, 2, 1], dtype=np.int32)
    plan = cloner.make_clone_plan(
        assets,
        [(0, 2), (1, 2), (0, 3), (1, 3)],
        8,
        clone_strategy=lambda weights, count: layout,
    )
    # Sources follow the first participating world prototype, then its first env.
    # Variant 1 also occupies env 0, but its earlier prototype first appears in env 2.
    paths = ["/World/envs/env_1/Object", "/World/envs/env_2/Object"]
    scene = _scene(plan, "{ENV_REGEX_NS}/Object", paths)
    groups = geometry_lookup(scene, "object")
    assert [(prim.GetPath(), ids) for prim, ids in groups] == [
        (paths[0], (1, 3, 4, 6)),
        (paths[1], (0, 2, 5, 7)),
    ]
    assert sorted(env_id for _, ids in groups for env_id in ids) == list(range(8))


def test_unknown_geometry_and_invalid_source_fail(geometry_lookup):
    plan = cloner.make_clone_plan([SimpleNamespace(prim_path="/World/envs/env_[^/]+/Object")], [(0,)], 2)
    scene = _scene(plan, "{ENV_REGEX_NS}/Other", [])
    with pytest.raises(
        AssertionError, match="must resolve one populated clone source per environment; resolved 0 of 2"
    ):
        geometry_lookup(scene, "object")
    scene.cfg.object.prim_path = "{ENV_REGEX_NS}/Object"
    with pytest.raises(AssertionError, match="is not a valid USD prim"):
        geometry_lookup(scene, "object")
