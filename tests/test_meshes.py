import trimesh
from jaxsim.parsers.rod import meshes


def test_mesh_wrapping_vertex_extraction():
    """Test the vertex extraction method on different meshes.
    1. A simple box
    2. A sphere
    """

    # Test 1: A simple box
    # First, create a box with origin at (0,0,0) and extents (3,3,3) -> points span from -1.5 to 1.5 on axis
    mesh = trimesh.creation.box(
        extents=[3.0, 3.0, 3.0],
    )
    points = meshes.MeshMapping.vertex_extraction(mesh)
    assert len(points) == len(mesh.vertices)

    # Test 2: A sphere
    # The sphere is centered at the origin and has a radius of 1.0
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    points = meshes.MeshMapping.vertex_extraction(mesh)
    assert len(points) == len(mesh.vertices)


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
    points = meshes.MeshMapping.aap(mesh, axis="x", aap_value=0.0, direction="higher")
    assert len(points) == len(mesh.vertices) // 2
    assert all(points[:, 0] > 0.0)

    # Test 1.2: Remove all points below y=0.0
    # Again, the expected result is that the number of points is halved
    points = meshes.MeshMapping.aap(mesh, axis="y", aap_value=0.0, direction="lower")
    assert len(points) == len(mesh.vertices) // 2
    assert all(points[:, 1] < 0.0)

    # Test 2: A sphere
    # The sphere is centered at the origin and has a radius of 1.0
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    # Remove all points above y=0.0
    points = meshes.MeshMapping.aap(mesh, axis="y", aap_value=0.0, direction="higher")
    assert all(points[:, 1] > 0.0)


def test_mesh_wrapping_points_over_axis():
    """Test the points over axis method on different meshes.
    1. A simple box
        1.1: Select 10 points from the lower end of the x-axis
        1.2: Select 10 points from the higher end of the y-axis
    2. A sphere
    """

    # Test 1.1: Remove 10 points from the lower end of the x-axis
    # First, create a box with origin at (0,0,0) and extents (3,3,3) -> points span from -1.5 to 1.5 on axis
    mesh = trimesh.creation.box(
        extents=[3.0, 3.0, 3.0],
    )
    points = meshes.MeshMapping.select_points_over_axis(
        mesh, axis="x", direction="lower", n=4
    )
    assert len(points) == 4
    assert all(points[:, 0] < 0.0)

    # Test 1.2: Select 10 points from the higher end of the y-axis
    points = meshes.MeshMapping.select_points_over_axis(
        mesh, axis="y", direction="higher", n=4
    )
    assert len(points) == 4
    assert all(points[:, 1] > 0.0)

    # Test 2: A sphere
    # The sphere is centered at the origin and has a radius of 1.0
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    sphere_n_vertices = len(mesh.vertices)

    # Select 10 points from the higher end of the z-axis
    points = meshes.MeshMapping.select_points_over_axis(
        mesh, axis="z", direction="higher", n=sphere_n_vertices // 2
    )
    assert len(points) == sphere_n_vertices // 2
    assert all(points[:, 2] >= 0.0)


def test_mesh_wrapping_object_mapping():
    """Test the object mapping method on different meshes.
    1. Subtract a box from a sphere
    2. Subtract a sphere from a bigger sphere
    3. Subtract a small box from a bigger box to remove the right-top corner of the first box
    """

    return  # Skip this test for now

    # Test 1: Subtract a box from a sphere
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    box = trimesh.creation.box(extents=[0.5, 0.5, 0.5])
    points = meshes.MeshMapping.object_mapping(sphere, box, method="subtraction")
    assert len(points) < len(sphere.vertices)

    # Test 2: Subtract a sphere from a bigger sphere
    sphere1 = trimesh.creation.icosphere(subdivisions=4, radius=1.5)
    sphere2 = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    points = meshes.MeshMapping.object_mapping(sphere1, sphere2, method="subtraction")
    assert len(points) < len(sphere1.vertices)
    assert len(points) > len(sphere2.vertices)

    # Test 3: Subtract a small box from a bigger box to remove the right-top corner of the first box
    box1 = trimesh.creation.box(extents=[3.0, 3.0, 3.0])
    box2 = trimesh.creation.box(
        extents=[1.0, 1.0, 1.0],
        transform=trimesh.transformations.translation_matrix([1.5, 1.5, 1.5]),
    )
    points = meshes.MeshMapping.object_mapping(box1, box2, method="subtraction")
    assert len(points) < len(box1)
    assert len(points) == 7
