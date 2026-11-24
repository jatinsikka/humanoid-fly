"""Thruster visuals parented to the robot's links, and the training-clip camera. Shared by both tasks.

Why parented shapes and not VisualizationMarkers: markers are re-placed from the robot state each render and draw a
frame (or more) behind the robot mesh, so at speed the flames and casings visibly trail the limbs. Shapes authored as
children of a link are moved by the renderer together with that link, so they cannot lag.

All shapes here are visual only (no collision, no mass): the thrust itself is applied in each env's _apply_action.

Flames: each flame is a "pivot" Xform under its link (translate = nozzle, orient = +z along the exhaust,
scale = (width, width, length)) with a unit cone below it. Flame length and direction can be animated per robot
with LiveFlames, which writes the pivot's scale/orient through USD. That needs every robot in USD
(scene.clone_in_fabric = False, as scripts/showcase.py sets). In training (4096 robots cloned in Fabric) the flames
stay at their authored length.
"""

from __future__ import annotations

import math
import re

import isaaclab.sim as sim_utils

FLAME_PIVOT = "thruster_flame"


# ---------------------------------------------------------------- small vector helpers (plain tuples, no conventions)
def xyzw_from_z_to(v) -> tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) rotating +z onto the direction v."""
    x, y, z = (float(c) for c in v)
    n = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    if z < -1.0 + 1e-9:  # opposite of +z: half turn about x
        return (1.0, 0.0, 0.0, 0.0)
    q = (-y, x, 0.0, 1.0 + z)  # half-angle form: (z_hat x v, 1 + z_hat . v), normalised
    s = math.sqrt(sum(c * c for c in q))
    return tuple(c / s for c in q)


def vadd(a, b, s: float = 1.0) -> tuple[float, float, float]:
    """a + s * b for 3-vectors."""
    return tuple(float(ai) + s * float(bi) for ai, bi in zip(a, b))


def unit(v) -> tuple[float, float, float]:
    n = math.sqrt(sum(float(c) ** 2 for c in v))
    return tuple(float(c) / n for c in v)


# ---------------------------------------------------------------- authoring (call in _setup_scene, before cloning)
def find_links(robot_root: str, names) -> dict | None:
    """{name: prim} for the named links under robot_root, or None if one is missing or not editable."""
    from pxr import Usd, UsdPhysics

    stage = sim_utils.get_current_stage()
    root = stage.GetPrimAtPath(robot_root)
    if not root.IsValid():
        print(f"[visuals] no prim at {robot_root}")
        return None
    # Newer URDF importers nest the links (pelvis_link/waist_roll_link/...) and give each link a mesh child with the
    # same name that is an instance (e.g. .../left_ankle_roll_link/left_ankle_roll_link). Several prims can share a
    # name, so prefer the rigid body, then the first (shallowest) editable prim.
    by_name = {}
    for p in Usd.PrimRange(root):
        if p.IsInstance() or p.IsInstanceProxy():
            continue
        rigid = p.HasAPI(UsdPhysics.RigidBodyAPI)
        if p.GetName() not in by_name or (rigid and not by_name[p.GetName()][1]):
            by_name[p.GetName()] = (p, rigid)
    out = {}
    for name in names:
        p = by_name.get(name, (None,))[0]
        if p is None:
            print(f"[visuals] link {name} under {robot_root} not found (or only as an instanceable mesh)")
            return None
        out[name] = p
    return out


def robot_roots(prim_path_expr: str) -> list[str]:
    """Robot prims that exist in USD for a regex prim path, sorted by env index."""
    paths = sim_utils.find_matching_prim_paths(prim_path_expr)

    def env_idx(p):
        m = re.search(r"env_(\d+)", p)
        return int(m.group(1)) if m else -1

    return sorted(paths, key=env_idx)


def add_flame(link_path: str, name: str, origin, exhaust, radius: float, length: float, width: float,
              color=(0.7, 0.9, 1.0), emissive=(0.55, 0.85, 1.0)) -> str:
    """Flame pivot + unit cone under link_path. The cone's apex points along `exhaust`, so it narrows away from the
    nozzle at `origin` (link frame). Returns the pivot path."""
    from pxr import Gf, UsdGeom

    stage = sim_utils.get_current_stage()
    path = f"{link_path}/{name}"
    xf = UsdGeom.Xform.Define(stage, path)
    x, y, z, w = xyzw_from_z_to(exhaust)
    xf.AddTranslateOp().Set(Gf.Vec3d(*origin))
    xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(width, width, length))
    cone = sim_utils.ConeCfg(
        radius=radius, height=1.0,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, emissive_color=emissive),
    )
    cone.func(f"{path}/cone", cone, translation=(0.0, 0.0, 0.5))  # unit cone from z=0 (nozzle) to z=1 (tip)
    return path


def add_cylinder(link_path: str, name: str, center, axis, radius: float, height: float, color, metallic=0.8):
    cfg = sim_utils.CylinderCfg(
        radius=radius, height=height,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=metallic, roughness=0.35),
    )
    cfg.func(f"{link_path}/{name}", cfg, translation=tuple(center), orientation=xyzw_from_z_to(axis))


def add_box(link_path: str, name: str, center, size, up, color, metallic=0.6):
    """Box whose local +z is `up` (link frame)."""
    cfg = sim_utils.CuboidCfg(
        size=tuple(size),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=metallic, roughness=0.5),
    )
    cfg.func(f"{link_path}/{name}", cfg, translation=tuple(center), orientation=xyzw_from_z_to(up))


# ---------------------------------------------------------------- animating flames (after cloning)
class LiveFlames:
    """Writes per-robot flame length/width (and optionally direction) through USD before each render.

    Only works when every robot is in USD (scene.clone_in_fabric=False). Otherwise `enabled` is False and the flames
    keep their authored size: animating only the robots that happen to be in USD would make env 0 look different
    from the rest of the grid.
    """

    def __init__(self, robot_prim_expr: str, rel_pivots: list[str], num_envs: int):
        from pxr import UsdGeom

        stage = sim_utils.get_current_stage()
        roots = robot_roots(robot_prim_expr)
        self.enabled = len(roots) == num_envs
        self._scale, self._orient = [], []
        if not self.enabled:
            print(f"[visuals] live flames off: {len(roots)} of {num_envs} robots are in USD "
                  "(set scene.clone_in_fabric=False to animate them); flames stay at their authored size")
            return
        for root in roots:
            s_row, o_row = [], []
            for rel in rel_pivots:
                ops = {op.GetOpType(): op for op in UsdGeom.Xformable(stage.GetPrimAtPath(f"{root}/{rel}")).GetOrderedXformOps()}
                s_row.append(ops[UsdGeom.XformOp.TypeScale].GetAttr())
                o_row.append(ops[UsdGeom.XformOp.TypeOrient].GetAttr())
            self._scale.append(s_row)
            self._orient.append(o_row)
        print(f"[visuals] live flames on for {len(roots)} robots x {len(rel_pivots)} thrusters")

    def update(self, lengths, widths, exhaust_dirs=None):
        """lengths, widths: (num_envs, k) array-likes; exhaust_dirs: optional (num_envs, k, 3) in the link frame."""
        if not self.enabled:
            return
        from pxr import Gf, Sdf

        lengths = lengths.tolist() if hasattr(lengths, "tolist") else lengths
        widths = widths.tolist() if hasattr(widths, "tolist") else widths
        dirs = exhaust_dirs.tolist() if hasattr(exhaust_dirs, "tolist") else exhaust_dirs
        with Sdf.ChangeBlock():
            for e, (s_row, o_row) in enumerate(zip(self._scale, self._orient)):
                for k, attr in enumerate(s_row):
                    w = widths[e][k]
                    attr.Set(Gf.Vec3f(w, w, max(lengths[e][k], 1e-3)))
                    if dirs is not None:
                        x, y, z, qw = xyzw_from_z_to(dirs[e][k])
                        o_row[k].Set(Gf.Quatf(qw, Gf.Vec3f(x, y, z)))


# ---------------------------------------------------------------- camera
def grid_camera(origins, back: float, height: float, ahead: float, look_height: float, hero: int = 0):
    """SO-101-style shot: camera just outside the hero env (env 0, at a corner of the grid), looking diagonally across
    the grid, so the hero robot is in front and the other envs fill the background. Returns (eye, lookat)."""
    o = origins.tolist() if hasattr(origins, "tolist") else list(origins)
    n = len(o)
    cx, cy = sum(p[0] for p in o) / n, sum(p[1] for p in o) / n
    hx, hy, hz = o[hero]
    dx, dy = hx - cx, hy - cy
    d = math.hypot(dx, dy)
    if d < 1e-6:  # single env: any direction
        dx, dy, d = 1.0, 1.0, math.sqrt(2.0)
    ux, uy = dx / d, dy / d
    eye = (hx + ux * back, hy + uy * back, hz + height)
    lookat = (hx - ux * ahead, hy - uy * ahead, hz + look_height)
    return eye, lookat
