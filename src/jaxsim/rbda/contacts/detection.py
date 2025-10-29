import jax
import jax.numpy as jnp

import jaxsim
import jaxsim.typing as jtp


def _contact_frame(normal: jtp.Vector, position: jtp.Vector) -> jtp.Matrix:
    """Create a contact frame with z-axis aligned with the contact normal."""
    n = normal / jaxsim.math.safe_norm(normal)

    t1_initial = jnp.array([1.0, 0.0, 0.0])

    t1 = t1_initial - jnp.dot(t1_initial, n) * n
    t1 = t1 / jaxsim.math.safe_norm(t1)
    t2 = jnp.cross(n, t1)

    R = jnp.stack([t1, t2, n], axis=1)

    return jnp.block(
        [
            [R[0, 0], R[0, 1], R[0, 2], position[0]],
            [R[1, 0], R[1, 1], R[1, 2], position[1]],
            [R[2, 0], R[2, 1], R[2, 2], position[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def sphere_plane(
    terrain: jaxsim.terrain.Terrain, size: jtp.Vector, W_H_L: jtp.Matrix
) -> tuple[jtp.Float, jtp.Matrix]:
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

    # Pad distance and transform to match expected output shapes.
    # and allow parallel evaluation of the collision types.
    distance = jnp.pad(jnp.array([distance]), (0, 2), mode="empty")
    W_H_C = jnp.pad(W_H_C[jnp.newaxis, ...], ((0, 2), (0, 0), (0, 0)), mode="empty")

    return distance, W_H_C


def box_plane(
    terrain: jaxsim.terrain.Terrain, size: jtp.Vector, W_H_L: jtp.Matrix
) -> tuple[jtp.Vector, jtp.Matrix]:
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

    # Project box z-axis on terrain normal and ensure direction away from plane
    sign = jnp.sign(R[:, 2])
    R_corrected = R.at[:, 2].set(R[:, 2] * sign)

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


def cylinder_plane(
    terrain: jaxsim.terrain.Terrain, size: jtp.Vector, W_H_L: jtp.Matrix
) -> tuple[jtp.Vector, jtp.Matrix]:
    """
    Return distances and contact frames of the 3 deepest points of a cylinder on terrain.

    Args:
        terrain: The terrain object.
        size: The size of the cylinder (radius, height).
        W_H_L: The collision shape transform in world coordinates.

    Returns:
        A tuple containing the distances from the cylinder to the plane and the pose transforms
        of the contact frames.
    """

    size = size.squeeze()
    r, half_h = size[0], size[1] * 0.5

    # Cylinder pose
    position = W_H_L[:3, 3]
    R = W_H_L[:3, :3]
    axis = R[:, 2]

    # Terrain data at cylinder XY
    h = terrain.height(position[0], position[1])
    n = terrain.normal(position[0], position[1])
    plane_position = jnp.array([position[0], position[1], h])

    # Project axis on normal and ensure direction away from plane
    prjaxis = jnp.dot(n, axis)
    sign = -jnp.sign(prjaxis + 1e-12)
    axis, prjaxis = axis * sign, prjaxis * sign

    # Distance from cylinder centre to plane along normal
    dist0 = jnp.dot(position - plane_position, n)

    # Remove component along normal from axis
    vec = axis * prjaxis - n
    len_vec = jnp.linalg.norm(vec)
    vec = jnp.where(
        len_vec < 1e-12,
        R[:, 0] * r,  # disk parallel to plane
        vec / len_vec * r,  # general case
    )

    # Project vec along normal
    prjvec = jnp.dot(vec, n)

    # Scale axis by half length
    ax_scaled = axis * half_h
    prjaxis_h = prjaxis * half_h

    # Sideways vector for 3-point support
    prjvec1 = -0.5 * prjvec
    vec1 = jnp.cross(vec, ax_scaled)
    vec1 = vec1 / (jnp.linalg.norm(vec1) + 1e-12) * r * jnp.sqrt(3.0) * 0.5

    # Distances of three candidate contacts:
    d1 = dist0 + prjaxis_h + prjvec
    d2 = dist0 + prjaxis_h + prjvec1
    dist = jnp.array([d1, d2, d2])

    # World position of candidates
    position_c = (
        position
        + ax_scaled
        + jnp.array(
            [
                vec - n * d1 * 0.5,
                vec1 + vec * 0.5 + n * d2 * 0.5,
                -vec1 + vec * 0.5 + n * d2 * 0.5,
            ]
        )
    )

    # Handle case in which the cylinder lies on the disks
    condition = jnp.abs(prjaxis) < 1e-3
    d3 = dist0 - prjaxis_h + prjvec
    dist = jnp.where(condition, dist.at[1].set(d3), dist)
    position_c = jnp.where(
        condition,
        position_c.at[1].set(position + vec - ax_scaled - n * d3 * 0.5),
        position_c,
    )

    # Build contact frames on the three candidate points
    W_H_C = jax.vmap(lambda p: _contact_frame(n, p))(position_c)

    return dist, W_H_C
