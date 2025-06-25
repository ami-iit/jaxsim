import jaxsim

def sphere_plane(
    terrain,
    size,
    center
):
    """
    Detects contacts between a sphere and a plane terrain.

    Args:
        terrain: The terrain object.
        size: The size of the sphere.
        center: The center of the sphere.

    Returns:
        A tuple containing the distance from the sphere to the plane and the pose transform
        of the contact frame.
    """
    normal = terrain.normal(center[:2])
    distance = jaxsim.math.safe_norm(center - terrain.height) @ normal - size[0]
    position = center - normal * (size[0] + 0.5 * distance)
    W_H_C = jaxsim.math.Transform.from_rotation_and_translation(
        rotation=jaxsim.math.Rotation.from_axis_angle(normal),
        translation=position,
    )
    return distance - size[0], normal