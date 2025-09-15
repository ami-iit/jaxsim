import jaxsim
import jaxsim.typing as jtp
import jax.numpy as jnp
import jax


def _contact_frame(normal, position):
    """Create a contact frame with z-axis aligned with the contact normal."""
    n = normal / jaxsim.math.safe_norm(normal)

    t1_initial = jnp.array([1.0, 0.0, 0.0])

    t1 = t1_initial - jnp.dot(t1_initial, n) * n
    t1 = t1 / jaxsim.math.safe_norm(t1)
    t2 = jnp.cross(n, t1)

    R = jnp.stack([t1, t2, n], axis=1)

    return jaxsim.math.Transform.from_rotation_and_translation(
        rotation=R,
        translation=position,
    )


def sphere_plane(terrain: jaxsim.terrain.Terrain, size: jtp.Vector, W_H_L: jtp.Matrix):
    """
    Detect contacts between a sphere and a plane terrain.

    Args:
        terrain: The terrain object.
        size: The size of the sphere.
        W_H_L: The collision shape transform in world coordinates.

    Returns:
        A tuple containing the distance from the sphere to the plane and the pose transform
        of the contact frame.
    """
    # Extract sphere center and radius.
    center = W_H_L[0:3, 3]
    radius = size[0]

    # Extract terrain properties at sphere center.
    x, y = center[0], center[1]

    normal = terrain.normal(x=x, y=y)
    height = terrain.height(x=x, y=y)

    distance = jnp.dot(center - height, normal) - radius

    position = center - radius * normal

    W_H_C = _contact_frame(normal, position)

    return distance, W_H_C


# TODO (flferretti): Keep only the SDF version?
def box_plane_sdf(terrain, size, W_H_L):
    """
    Return distances and contact frames of the 3 deepest corners of a box on terrain using SDF.
    Fully vectorized, works for any box orientation.
    """
    half_size = size.squeeze() / 2

    R = W_H_L[:3, :3]
    center = W_H_L[:3, 3]

    # Generate all 8 corners using meshgrid
    sx = jnp.array([-half_size[0], half_size[0]])
    sy = jnp.array([-half_size[1], half_size[1]])
    sz = jnp.array([-half_size[2], half_size[2]])
    xs, ys, zs = jnp.meshgrid(sx, sy, sz, indexing="ij")
    corners_local = jnp.stack(
        [xs.ravel(), ys.ravel(), zs.ravel()], axis=1
    )  # shape (8,3)

    box_z_world = R[:, 2]
    flip_sign = jnp.sign(box_z_world)
    R_corrected = R.at[:, 2].set(R[:, 2] * flip_sign)  # flip z-axis if needed

    # Transform to world frame
    corners_world = center + corners_local @ R_corrected.T

    # Vectorized terrain height and normal using vmap
    terrain_height_vmap = jax.vmap(lambda p: terrain.height(p[0], p[1]))
    terrain_normal_vmap = jax.vmap(lambda p: terrain.normal(p[0], p[1]))

    terrain_heights = terrain_height_vmap(corners_world)
    terrain_points = jnp.stack(
        [corners_world[:, 0], corners_world[:, 1], terrain_heights], axis=1
    )

    normals = terrain_normal_vmap(corners_world)

    # Distances along terrain normal
    distances = jnp.einsum("ij,ij->i", corners_world - terrain_points, normals)

    # Pick 3 closest points using top_k
    _, topk_idx = jax.lax.top_k(-distances, 3)
    contact_points = corners_world[topk_idx]
    contact_normals = normals[topk_idx]

    # Compute contact frames using vmap
    W_H_C = jax.vmap(lambda p, n: _contact_frame(n, p))(contact_points, contact_normals)

    # Distances along terrain normal for the selected points
    distances_top3 = distances[topk_idx]

    return distances_top3, W_H_C


def box_plane(
    terrain: jaxsim.terrain.Terrain,
    size: jtp.Vector,
    W_H_L: jtp.Matrix,
):
    """
    Detect contacts between a box and a plane terrain.
    Finds the actual contact point on the box surface (vertex, edge, or face).

    Args:
        terrain: The terrain object with _height(x, y) method and normal(x, y) method.
        size: A 3D vector [width, height, depth] representing the box dimensions from center.
        W_H_L: The collision shape transform in world coordinates.

    Returns:
        A tuple containing the distance from the box to the plane and the pose transform
        of the contact frame.
    """
    half_size = size.squeeze() / 2
    center = W_H_L[:3, 3]
    R = W_H_L[:3, :3]

    # Transform terrain normal at box center into world coordinates
    normal = terrain.normal(center[0], center[1])

    # Find the box vertex furthest in the opposite direction of terrain normal
    local_normal = R.T @ normal
    support_local = -half_size * jnp.sign(local_normal)

    # Vertex in world coordinates
    support_world = center + R @ support_local

    # Terrain point and distance
    terrain_z = terrain.height(support_world[0], support_world[1])
    terrain_point = jnp.array([support_world[0], support_world[1], terrain_z])
    distance = jnp.dot(support_world - terrain_point, normal)

    # Contact frame
    contact_point = support_world - distance * normal
    W_H_C = _contact_frame(normal, contact_point)

    return distance, W_H_C


def cylinder_plane(
    terrain: jaxsim.terrain.Terrain,
    size: jtp.Vector,
    W_H_L: jtp.Matrix,
):
    """
    Compact cylinder-plane contact detection returning distance and SE(3) contact frame.
    """
    size = size.squeeze()
    radius, half_height = size[0], size[1] / 2.0

    # Cylinder center and axis
    center = W_H_L[0:3, 3]
    axis = W_H_L[0:3, 2] / jnp.linalg.norm(W_H_L[0:3, 2])

    # Terrain properties at center
    n = terrain.normal(center[0], center[1])

    # Axis projection and perpendicular direction
    axis_dot_n = jnp.dot(axis, n)
    perp = jnp.cross(axis, n)
    perp_norm = jnp.linalg.norm(perp)
    perp = jnp.where(perp_norm > 1e-6, perp / perp_norm, W_H_L[0:3, 0])

    # Three potential contact points
    cap_offset = axis * half_height * jnp.sign(axis_dot_n)
    edge_offset = perp * radius

    contacts = jnp.array(
        [
            center + cap_offset,
            center + edge_offset + axis * half_height,
            center + edge_offset - axis * half_height,
        ]
    )

    # Vectorized terrain height computation
    terrain_heights = jax.vmap(terrain.height)(contacts[:, 0], contacts[:, 1])
    terrain_points = jnp.column_stack([contacts[:, :2], terrain_heights])

    # Compute distances
    distances = jnp.sum((contacts - terrain_points) * n, axis=1)

    # Select best contact based on axis alignment and minimum distance
    abs_dot = jnp.abs(axis_dot_n)
    weights = jnp.where(
        abs_dot > 0.8,  # Face contact - use cap
        jnp.array([1.0, 0.0, 0.0]),
        jnp.where(
            abs_dot < 0.3,  # Edge contact - use minimum edge distance
            jnp.array([0.0, 1.0, 1.0]),
            jnp.array([1.0, 0.0, 0.0]),  # Corner contact - use cap
        ),
    )

    # Find minimum valid distance
    valid_distances = jnp.where(weights > 0, distances, jnp.inf)
    min_idx = jnp.argmin(valid_distances)

    distance = distances[min_idx]
    contact_pos = contacts[min_idx]

    W_H_C = _contact_frame(n, contact_pos)

    return distance, W_H_C
