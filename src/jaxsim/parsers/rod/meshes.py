from collections.abc import Sequence

import numpy as np
import rod
import trimesh

from jaxsim import logging


def parse_object_mapping_object(obj) -> trimesh.Trimesh:
    if isinstance(obj, trimesh.Trimesh):
        return obj
    elif isinstance(obj, dict):
        if "type" not in obj:
            raise ValueError("Object type not specified")
        if obj["type"] == "box":
            if "extents" not in obj:
                raise ValueError("Box extents not specified")
            return trimesh.creation.box(extents=obj["extents"])
        elif obj["type"] == "sphere":
            if "radius" not in obj:
                raise ValueError("Sphere radius not specified")
            return trimesh.creation.icosphere(subdivisions=4, radius=obj["radius"])
        else:
            raise ValueError(f"Invalid object type {obj['type']}")
    elif isinstance(obj, rod.builder.primitive_builder.PrimitiveBuilder):
        raise NotImplementedError("PrimitiveBuilder not implemented")
    else:
        raise ValueError("Invalid object type")


def extract_points_vertices(mesh) -> np.ndarray:
    return mesh.vertices


def extract_points_random_surface_sampling(mesh, n: int = -1) -> np.ndarray:
    if n > 0 and n <= len(mesh.vertices):
        return mesh.sample(n)
    else:
        if n != -1:
            logging.warning(
                "Invalid number of points for random surface sampling. Defaulting to all vertices"
            )
        return mesh.vertices


def extract_points_uniform_surface_sampling(mesh, n: int = -1) -> np.ndarray:
    if n > 0 and n <= len(mesh.vertices):
        return trimesh.sample.sample_surface_even(mesh=mesh, count=n)
    else:
        if n != -1:
            logging.warning(
                "Invalid number of points for uniform surface sampling. Defaulting to all vertices"
            )
        return mesh.vertices


def extract_points_select_points_over_axis(
    mesh, axis: str, direction: str, n: int
) -> np.ndarray:
    valid_dirs = ["higher", "lower"]
    if direction not in valid_dirs:
        raise ValueError(f"Invalid direction. Valid directions are {valid_dirs}")

    arr = mesh.vertices

    index = 0 if axis == "x" else 1 if axis == "y" else 2
    # Sort the array in ascending order
    sorted_arr = arr[arr[:, index].argsort()]

    if direction == "lower":
        # Select first N points
        points = sorted_arr[:n]
    elif direction == "higher":
        # Select last N points
        points = sorted_arr[-n:]
    else:
        raise ValueError(
            f"Invalid direction {direction} for SelectPointsOverAxis method"
        )

    return points


def extract_points_app(
    mesh: trimesh.Trimesh,
    axis: str,
    upper: float | None = None,
    lower: float | None = None,
) -> np.ndarray:
    # Check bounds
    upper = upper if upper is not None else np.inf
    lower = lower if lower is not None else -np.inf
    assert lower < upper, "Invalid bounds for axis-aligned plane"

    # Logic
    valid_axes = {"x": 0, "y": 1, "z": 2}

    points = mesh.vertices[
        (mesh.vertices[:, valid_axes[axis]] >= lower)
        & (mesh.vertices[:, valid_axes[axis]] <= upper)
    ]

    return points


def extract_points_object_mapping(
    mesh: trimesh.Trimesh,
    objs: Sequence[trimesh.Trimesh | dict],
    method: str = "subtract",
) -> np.ndarray:
    valid_methods = ["subtract", "intersect"]
    if method not in valid_methods:
        raise ValueError(f"Invalid method {method} for object mapping")
    if len(objs) == 0:
        return mesh.vertices
    if method == "subtract":
        for obj in objs:
            mesh = mesh.difference(obj)
    elif method == "intersect":
        for obj in objs:
            mesh = mesh.intersection(obj)
    else:
        raise ValueError(f"Invalid method {method} for object mapping")

    return mesh.vertices
