import pytest
import tempfile
import trimesh
from jaxsim.parsers.rod import meshes


def test_mesh_wrapping_aap():
    """Test the AAP wrapping method on different meshes.
    1. A simple box
        1.1: Remove all points above x=0.0
        1.2: Remove all points below y=0.0
    2. A sphere
    """

    # Test 1.1: Remove all points above x=0.0
    # The expected result is that the number of points is halved
    # First, create a box with origin at (0,0,0) and extents (3,3,3) -> points span from -1.5 to 1.5 on axis
    mesh = trimesh.creation.box(
        extents=[3.0, 3.0, 3.0],
    )
    points = meshes.extract_points_aap(
        mesh, aap_axis="x", aap_value=0.0, aap_direction="higher"
    )
    assert len(points) == len(mesh.vertices) // 2
    assert all(points[:, 0] > 0.0)

    # Test 1.2: Remove all points below y=0.0
    # Again, the expected result is that the number of points is halved
    points = meshes.extract_points_aap(
        mesh, aap_axis="y", aap_value=0.0, aap_direction="lower"
    )
    assert len(points) == len(mesh.vertices) // 2
    assert all(points[:, 1] < 0.0)

    # Test 2: A sphere
    # The sphere is centered at the origin and has a radius of 1.0
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    # Remove all points above y=0.0
    points = meshes.extract_points_aap(
        mesh, aap_axis="y", aap_value=0.0, aap_direction="higher"
    )
    assert all(points[:, 1] > 0.0)
