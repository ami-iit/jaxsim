import numpy as np
import trimesh

VALID_AXIS = {"x": 0, "y": 1, "z": 2}


def extract_points_vertices(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Extract the vertices of a mesh as points.
    """
    return mesh.vertices


def extract_points_random_surface_sampling(mesh: trimesh.Trimesh, n) -> np.ndarray:
    """
    Extract N random points from the surface of a mesh.

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
    Extract N uniformly sampled points from the surface of a mesh.

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
    Extract N points from a mesh along a specified axis. The points are selected based on their position along the axis.

    Args:
        mesh: The mesh from which to extract points.
        axis: The axis along which to extract points.
        direction: The direction along the axis from which to extract points. Valid values are "higher" and "lower".
        n: The number of points to extract.

    Returns:
        The extracted points (N x 3 array).
    """

    dirs = {"higher": np.s_[-n:], "lower": np.s_[:n]}
    arr = mesh.vertices

    # Sort rows lexicographically first, then columnar.
    arr.sort(axis=0)
    sorted_arr = arr[dirs[direction]]
    return sorted_arr


def extract_points_aap(
    mesh: trimesh.Trimesh,
    axis: str,
    upper: float | None = None,
    lower: float | None = None,
) -> np.ndarray:
    """
    Extract points from a mesh along a specified axis within a specified range. The points are selected based on their position along the axis.

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

    # Check bounds.
    upper = upper if upper is not None else np.inf
    lower = lower if lower is not None else -np.inf
    assert lower < upper, "Invalid bounds for axis-aligned plane"

    # Logic.
    points = mesh.vertices[
        (mesh.vertices[:, VALID_AXIS[axis]] >= lower)
        & (mesh.vertices[:, VALID_AXIS[axis]] <= upper)
    ]

    return points
