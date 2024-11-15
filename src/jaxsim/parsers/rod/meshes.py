from collections.abc import Sequence

import numpy as np
import trimesh

from jaxsim import logging

VALID_AXIS = {"x": 0, "y": 1, "z": 2}


def parse_object_mapping_object(obj: trimesh.Trimesh | dict) -> trimesh.Trimesh:
    if isinstance(obj, trimesh.Trimesh):
        return obj
    elif isinstance(obj, dict):
        if obj["type"] == "box":
            return trimesh.creation.box(extents=obj["extents"])
        elif obj["type"] == "sphere":
            return trimesh.creation.icosphere(subdivisions=4, radius=obj["radius"])
        else:
            raise ValueError(f"Invalid object type {obj['type']}")
    else:
        raise ValueError("Invalid object type")


def extract_points_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Extracts the vertices of a mesh as points.
    """
    return mesh.vertices


def extract_points_random_surface_sampling(mesh: trimesh.Trimesh, n) -> np.ndarray:
    """
    Extracts N random points from the surface of a mesh.

    Args:
        mesh: The mesh from which to extract points.
        n: The number of points to extract.

    Returns:
        The extracted points (N x 3 array).
    """

    return mesh.sample(n)


def extract_points_uniform_surface_sampling(
    mesh: trimesh.Trimesh, n: int
) -> np.ndarray:
    """
    Extracts N uniformly sampled points from the surface of a mesh.

    Args:
        mesh: The mesh from which to extract points.
        n: The number of points to extract.

    Returns:
        The extracted points (N x 3 array).
    """

    return trimesh.sample.sample_surface_even(mesh=mesh, count=n)[0]


def extract_points_select_points_over_axis(
    mesh: trimesh.Trimesh, axis: str, direction: str, n: int
) -> np.ndarray:
    """
    Extracts N points from a mesh along a specified axis. The points are selected based on their position along the axis.

    Args:
        mesh: The mesh from which to extract points.
        axis: The axis along which to extract points.
        direction: The direction along the axis from which to extract points. Valid values are "higher" and "lower".
        n: The number of points to extract.

    Returns:
        The extracted points (N x 3 array).
    """

    dirs = {"higher": np.s_[:n], "lower": np.s_[-n:]}
    arr = mesh.vertices
    index = dict(zip(("x", "y", "z"), np.arange(3), strict=False))

    # Sort the array in ascending order
    arr.sort(axis=index[axis])
    return arr[dirs[direction]]


def extract_points_aap(
    mesh: trimesh.Trimesh,
    axis: str,
    upper: float | None = None,
    lower: float | None = None,
) -> np.ndarray:
    """
    Extracts points from a mesh along a specified axis within a specified range. The points are selected based on their position along the axis.

    Args:
        mesh: The mesh from which to extract points.
        axis: The axis along which to extract points.
        upper: The upper bound of the range.
        lower: The lower bound of the range.

    Returns:
        The extracted points (N x 3 array).

    Raises:
        AssertionError: If the lower bound is greater than the upper bound.
    """

    # Check bounds
    upper = upper if upper is not None else np.inf
    lower = lower if lower is not None else -np.inf
    assert lower < upper, "Invalid bounds for axis-aligned plane"

    # Logic

    points = mesh.vertices[
        (mesh.vertices[:, VALID_AXIS[axis]] >= lower)
        & (mesh.vertices[:, VALID_AXIS[axis]] <= upper)
    ]

    return points


def extract_points_object_mapping(
    mesh: trimesh.Trimesh,
    objs: Sequence[trimesh.Trimesh | dict],
    method: str = "subtract",
) -> np.ndarray:
    """
    Extracts points from a mesh by mapping objects onto it.

    Args:
        mesh: The mesh from which to extract points.
        objs: The objects to map onto the mesh.
        method: The method to use for object mapping. Valid values are "subtract" and "intersect".

    Returns:
        The extracted points (N x 3 array).

    Raises:
        ValueError: If an invalid method is provided.
    """

    valid_methods = {
        "subtract": trimesh.Trimesh.difference,
        "intersect": trimesh.Trimesh.intersection,
    }
    if method not in valid_methods:
        raise ValueError(f"Invalid method {method} for object mapping")
    if len(objs) == 0:
        logging.warning(
            "No objects provided for object mapping, returning original mesh"
        )
        return mesh.vertices

    # Parse objects
    for obj in objs:
        obj = parse_object_mapping_object(obj)
        mesh = valid_methods[method](mesh, obj)

    return mesh.vertices
