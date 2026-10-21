# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Uneven-floor terrain for the lacrosse scoop: a polygon (triangle) mesh built from one global height map.

The terrain is a grid of tiles, ``levels`` rows (difficulty, from flat to the roughest setting) by ``tiles`` columns.
Each tile is a smooth random bump field plus a planar tilt in a random direction, both scaled with the tile's level.
Keeping the whole floor as one regular height map lets the environment look up the exact floor height under the ball
or the wheels on the GPU (:class:`HeightMap`), with the same triangulation as the collision mesh.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import trimesh
from scipy.ndimage import gaussian_filter


def make_rough_floor(
    levels: int,
    tiles: int,
    tile_size: tuple[float, float],
    spacing: float,
    bump_rms_max: float,
    bump_corr: float,
    tilt_max_deg: float,
    seed: int = 0,
) -> dict:
    """Generate the floor height map.

    Args:
        levels: Number of difficulty levels (tile rows along x). Level ``k`` uses ``k / (levels - 1)`` of the maxima.
        tiles: Number of tiles per level (along y).
        tile_size: Tile length along x and width along y [m].
        spacing: Height-map grid spacing [m].
        bump_rms_max: RMS bump height at the top level [m].
        bump_corr: Bump correlation length (Gaussian smoothing sigma) [m].
        tilt_max_deg: Tile tilt at the top level [deg].
        seed: Random seed.

    Returns:
        ``heights`` (nx, ny) [m] with ``x0``, ``y0`` (world position of heights[0, 0]) and ``spacing`` [m], the tile
        ``centers`` (levels, tiles, 3) [m] (floor height included) and the per-tile ``level_frac`` (levels, tiles).
    """
    rng = np.random.default_rng(seed)
    nx_t, ny_t = round(tile_size[0] / spacing), round(tile_size[1] / spacing)
    heights = np.zeros((levels * nx_t + 1, tiles * ny_t + 1))
    x0, y0 = -0.5 * levels * tile_size[0], -0.5 * tiles * tile_size[1]
    centers = np.zeros((levels, tiles, 3))
    level_frac = np.zeros((levels, tiles))
    lx = (np.arange(nx_t + 1) - 0.5 * nx_t) * spacing
    ly = (np.arange(ny_t + 1) - 0.5 * ny_t) * spacing
    gx, gy = np.meshgrid(lx, ly, indexing="ij")
    for i in range(levels):
        frac = i / max(levels - 1, 1)
        for j in range(tiles):
            noise = gaussian_filter(rng.standard_normal(gx.shape), sigma=bump_corr / spacing, mode="wrap")
            bumps = noise / max(noise.std(), 1e-12) * frac * bump_rms_max
            phi = rng.uniform(-math.pi, math.pi)
            slope = math.tan(math.radians(frac * tilt_max_deg))
            tile = bumps + slope * (math.cos(phi) * gx + math.sin(phi) * gy)
            tile -= tile[nx_t // 2, ny_t // 2]  # tile center at height 0
            heights[i * nx_t : (i + 1) * nx_t + 1, j * ny_t : (j + 1) * ny_t + 1] = tile
            centers[i, j] = (x0 + (i + 0.5) * tile_size[0], y0 + (j + 0.5) * tile_size[1], 0.0)
            level_frac[i, j] = frac
    return dict(heights=heights, x0=x0, y0=y0, spacing=spacing, centers=centers, level_frac=level_frac)


def height_map_to_mesh(heights: np.ndarray, x0: float, y0: float, spacing: float) -> trimesh.Trimesh:
    """Triangulate a height map; each grid cell is split along its (i+1, j)-(i, j+1) diagonal."""
    nx, ny = heights.shape
    gx, gy = np.meshgrid(x0 + np.arange(nx) * spacing, y0 + np.arange(ny) * spacing, indexing="ij")
    vertices = np.stack([gx.ravel(), gy.ravel(), heights.ravel()], axis=-1)
    idx = np.arange(nx * ny).reshape(nx, ny)
    v00, v10, v01, v11 = idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:], idx[1:, 1:]
    # counter-clockwise seen from above (+z normals)
    faces = np.concatenate(
        [np.stack([v00, v10, v01], -1).reshape(-1, 3), np.stack([v10, v11, v01], -1).reshape(-1, 3)], axis=0
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


class HeightMap:
    """Exact floor height of the triangulated height map, evaluated on the GPU."""

    def __init__(self, heights: np.ndarray, x0: float, y0: float, spacing: float, device: str):
        self.h = torch.tensor(heights, dtype=torch.float32, device=device)
        self.x0, self.y0, self.spacing = x0, y0, spacing

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Floor height [m] at world (x, y) [m]; points outside the map are clamped to its border."""
        nx, ny = self.h.shape
        u = ((x - self.x0) / self.spacing).clamp(0.0, nx - 1.0001)
        v = ((y - self.y0) / self.spacing).clamp(0.0, ny - 1.0001)
        i, j = u.floor().long(), v.floor().long()
        fx, fy = u - i, v - j
        h00, h10 = self.h[i, j], self.h[i + 1, j]
        h01, h11 = self.h[i, j + 1], self.h[i + 1, j + 1]
        lower = h00 + fx * (h10 - h00) + fy * (h01 - h00)
        upper = h11 + (1.0 - fx) * (h01 - h11) + (1.0 - fy) * (h10 - h11)
        return torch.where(fx + fy <= 1.0, lower, upper)
