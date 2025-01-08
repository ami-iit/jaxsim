import trimesh

from jaxsim.parsers.rod import meshes


def test_mesh_wrapping_vertex_extraction():
    """
    Test the vertex extraction method on different meshes.

    1. A simple box.
    2. A sphere.
    """

    # Test 1: A simple box.
    #     First, create a box with origin at (0,0,0) and extents (3,3,3),
    #     i.e. points span from -1.5 to 1.5 on the axis.
    mesh = trimesh.creation.box(
        extents=[3.0, 3.0, 3.0],
    )
    points = meshes.extract_points_vertices(mesh=mesh)
    assert len(points) == len(mesh.vertices)

    # Test 2: A sphere.
    #     The sphere is centered at the origin and has a radius of 1.0.
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    points = meshes.extract_points_vertices(mesh=mesh)
    assert len(points) == len(mesh.vertices)


def test_mesh_wrapping_aap():
    """
    Test the AAP wrapping method on different meshes.

    1. A simple box
        1.1: Remove all points above x=0.0
        1.2: Remove all points below y=0.0
    2. A sphere
    """

    # Test 1.1: Remove all points above x=0.0.
    #     The expected result is that the number of points is halved.
    #     First, create a box with origin at (0,0,0) and extents (3,3,3),
    #     i.e. points span from -1.5 to 1.5 on the axis.
    mesh = trimesh.creation.box(extents=[3.0, 3.0, 3.0])
    points = meshes.extract_points_aap(mesh=mesh, axis="x", lower=0.0)
    assert len(points) == len(mesh.vertices) // 2
    assert all(points[:, 0] > 0.0)

    # Test 1.2: Remove all points below y=0.0.
    #     The expected result is that the number of points is halved.
    points = meshes.extract_points_aap(mesh=mesh, axis="y", upper=0.0)
    assert len(points) == len(mesh.vertices) // 2
    assert all(points[:, 1] < 0.0)

    # Test 2: A sphere.
    #     The sphere is centered at the origin and has a radius of 1.0.
    #     Points are expected to be halved.
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)

    # Remove all points above y=0.0.
    points = meshes.extract_points_aap(mesh=mesh, axis="y", lower=0.0)
    assert all(points[:, 1] >= 0.0)
    assert len(points) < len(mesh.vertices)


def test_mesh_wrapping_points_over_axis():
    """
    Test the points over axis method on different meshes.

    1. A simple box
        1.1: Select 10 points from the lower end of the x-axis
        1.2: Select 10 points from the higher end of the y-axis
    2. A sphere
    """

    # Test 1.1: Remove 10 points from the lower end of the x-axis.
    #     First, create a box with origin at (0,0,0) and extents (3,3,3),
    #     i.e. points span from -1.5 to 1.5 on the axis.
    mesh = trimesh.creation.box(extents=[3.0, 3.0, 3.0])
    points = meshes.extract_points_select_points_over_axis(
        mesh=mesh, axis="x", direction="lower", n=4
    )
    assert len(points) == 4
    assert all(points[:, 0] < 0.0)

    # Test 1.2: Select 10 points from the higher end of the y-axis.
    points = meshes.extract_points_select_points_over_axis(
        mesh=mesh, axis="y", direction="higher", n=4
    )
    assert len(points) == 4
    assert all(points[:, 1] > 0.0)

    # Test 2: A sphere.
    #     The sphere is centered at the origin and has a radius of 1.0.
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    sphere_n_vertices = len(mesh.vertices)

    # Select 10 points from the higher end of the z-axis.
    points = meshes.extract_points_select_points_over_axis(
        mesh=mesh, axis="z", direction="higher", n=sphere_n_vertices // 2
    )
    assert len(points) == sphere_n_vertices // 2
    assert all(points[:, 2] >= 0.0)
