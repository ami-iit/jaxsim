import os

if "ROD_LOGGING_LEVEL" not in os.environ:
    os.environ["ROD_LOGGING_LEVEL"] = "WARNING"

import numpy as np
import rod.kinematics.tree_transforms
from rod.builder import primitives

if __name__ == "__main__":

    # ================
    # Model parameters
    # ================

    # Rail parameters.
    rail_height = 1.2
    rail_length = 5.0
    rail_radius = 0.005
    rail_mass = 5.0

    # Cart parameters.
    cart_mass = 1.0
    cart_size = (0.1, 0.2, 0.05)

    # Pole parameters.
    pole_mass = 0.5
    pole_length = 1.0
    pole_radius = 0.005

    # ========================
    # Create the link builders
    # ========================

    rail_builder = primitives.CylinderBuilder(
        name="rail",
        mass=rail_mass,
        radius=rail_radius,
        length=rail_length,
    )

    cart_builder = primitives.BoxBuilder(
        name="cart",
        mass=cart_mass,
        x=cart_size[0],
        y=cart_size[1],
        z=cart_size[2],
    )

    pole_builder = primitives.CylinderBuilder(
        name="pole",
        mass=pole_mass,
        radius=pole_radius,
        length=pole_length,
    )

    # =================
    # Create the joints
    # =================

    world_to_rail = rod.Joint(
        name="world_to_rail",
        type="fixed",
        parent="world",
        child=rail_builder.name,
        pose=primitives.PrimitiveBuilder.build_pose(
            relative_to="world",
        ),
    )

    linear = rod.Joint(
        name="linear",
        type="prismatic",
        parent=rail_builder.name,
        child=cart_builder.name,
        pose=primitives.PrimitiveBuilder.build_pose(
            relative_to=rail_builder.name,
            pos=np.array([0, 0, rail_height]),
        ),
        axis=rod.Axis(
            xyz=rod.Xyz(xyz=[0, 1, 0]),
            limit=rod.Limit(
                upper=(rail_length / 2 - cart_size[1] / 2),
                lower=-(rail_length / 2 - cart_size[1] / 2),
                effort=500.0,
                velocity=10.0,
            ),
        ),
    )

    pivot = rod.Joint(
        name="pivot",
        type="continuous",
        parent=cart_builder.name,
        child=pole_builder.name,
        pose=primitives.PrimitiveBuilder.build_pose(
            relative_to=cart_builder.name,
        ),
        axis=rod.Axis(
            xyz=rod.Xyz(xyz=[1, 0, 0]),
            limit=rod.Limit(),
        ),
    )

    # ================
    # Create the links
    # ================

    rail_elements_pose = primitives.PrimitiveBuilder.build_pose(
        pos=np.array([0, 0, rail_height]),
        rpy=np.array([np.pi / 2, 0, 0]),
    )

    rail = (
        rail_builder.build_link(
            name=rail_builder.name,
            pose=primitives.PrimitiveBuilder.build_pose(
                relative_to=world_to_rail.name,
            ),
        )
        .add_inertial(pose=rail_elements_pose)
        .add_visual(pose=rail_elements_pose)
        .add_collision(pose=rail_elements_pose)
        .build()
    )

    cart = (
        cart_builder.build_link(
            name=cart_builder.name,
            pose=primitives.PrimitiveBuilder.build_pose(relative_to=linear.name),
        )
        .add_inertial()
        .add_visual()
        .add_collision()
        .build()
    )

    pole_elements_pose = primitives.PrimitiveBuilder.build_pose(
        pos=np.array([0, 0, pole_length / 2]),
    )

    pole = (
        pole_builder.build_link(
            name=pole_builder.name,
            pose=primitives.PrimitiveBuilder.build_pose(
                relative_to=pivot.name,
            ),
        )
        .add_inertial(pose=pole_elements_pose)
        .add_visual(pose=pole_elements_pose)
        .add_collision(pose=pole_elements_pose)
        .build()
    )

    # ===========
    # Build model
    # ===========

    # Create ROD in-memory model.
    model = rod.Model(
        name="cartpole",
        canonical_link=rail.name,
        link=[
            rail,
            cart,
            pole,
        ],
        joint=[
            world_to_rail,
            linear,
            pivot,
        ],
    )

    # Update the pose elements to be closer to those expected in URDF.
    model.switch_frame_convention(
        frame_convention=rod.FrameConvention.Urdf, explicit_frames=True
    )

    # ==============
    # Get SDF string
    # ==============

    # Create the top-level SDF object.
    sdf = rod.Sdf(version="1.10", model=model)

    # Generate the SDF string.
    # sdf_string = sdf.serialize(pretty=True, validate=True)

    # ===============
    # Get URDF string
    # ===============

    import rod.urdf.exporter

    # Convert the SDF to URDF.
    urdf_string = rod.urdf.exporter.UrdfExporter(
        pretty=True, indent="    "
    ).to_urdf_string(sdf=sdf)

    # Print the URDF string.
    print(urdf_string)
