import jaxsim
import jaxsim.typing as jtp

def sphere_plane(
    terrain: jaxsim.terrain.Terrain,
    size: jtp.Vector,
    center: jtp.Vector,
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

def box_plane(
    terrain: jaxsim.terrain.Terrain,
    size: jtp.Vector,
    center: jtp.Vector,
):
    """
    Detects contacts between a box and a plane terrain.

    Args:
        terrain: The terrain object.
        size: The size of the box.
        center: The center of the box.

    Returns:
        A tuple containing the distance from the box to the plane and the pose transform
        of the contact frame.
    """
    # normal = terrain.normal(center[:2])
    # distance = jaxsim.math.safe_norm(jnp.max(center - terrain.height, axis=0)) @ normal - size[0]


    #   return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
    # W_H_C = jaxsim.math.Transform.from_rotation_and_translation(
    #     rotation=jaxsim.math.Rotation.from_axis_angle(normal),
    #     translation=position,
    # )
    # return distance - size[0], normal, W_H_C