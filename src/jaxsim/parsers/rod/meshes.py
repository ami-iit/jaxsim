import trimesh
import numpy as np


def extract_points_vertex_extraction(mesh: trimesh.Trimesh) -> np.ndarray:
    """Extracts the points of a mesh using the vertices of the mesh as colliders.

    Args:
        mesh: The mesh to extract the points from.

    Returns:
        The points of the mesh.
    """
    return mesh.vertices


def extract_points_random_surface_sampling(
    mesh: trimesh.Trimesh, num_points: int
) -> np.ndarray:
    """Extracts the points of a mesh by sampling the surface of the mesh randomly.

    Args:
        mesh: The mesh to extract the points from.
        num_points: The number of points to sample.

    Returns:
        The points of the mesh.
    """
    return mesh.sample(num_points)


def extract_points_uniform_surface_sampling(
    mesh: trimesh.Trimesh, num_points: int
) -> np.ndarray:
    """Extracts the points of a mesh by sampling the surface of the mesh uniformly.

    Args:
        mesh: The mesh to extract the points from.
        num_points: The number of points to sample.

    Returns:
        The points of the mesh.
    """
    return trimesh.sample.sample_surface_even(mesh=mesh, count=num_points)


def extract_points_aap(
    mesh: trimesh.Trimesh,
    aap_axis: str,
    aap_value: float,
    aap_direction: str,
) -> np.ndarray:
    """Extracts the points of a mesh that are on one side of an axis-aligned plane (AAP).

    Args:
        mesh: The mesh to extract the points from.
        aap_axis: The axis of the AAP.
        aap_value: The value of the AAP.
        aap_direction: The direction of the AAP.

    Returns:
        The points of the mesh that are on one side of the AAP.
    """
    if aap_direction == "higher":
        aap_operator = np.greater
    elif aap_direction == "lower":
        aap_operator = np.less
    else:
        raise ValueError("Invalid direction for axis-aligned plane")

    if aap_axis == "x":
        points = mesh.vertices[aap_operator(mesh.vertices[:, 0], aap_value)]
    elif aap_axis == "y":
        points = mesh.vertices[aap_operator(mesh.vertices[:, 1], aap_value)]
    elif aap_axis == "z":
        points = mesh.vertices[aap_operator(mesh.vertices[:, 2], aap_value)]
    else:
        raise ValueError("Invalid axis for axis-aligned plane")

    return points
