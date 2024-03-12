from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import jax_dataclasses
from jax_dataclasses import Static

import jaxsim.typing as jtp
from jaxsim.parsers.descriptions import ModelDescription


@jax_dataclasses.pytree_dataclass
class GroundContact:
    """
    Class storing the collidable points of a robot model.

    This class is used to store and manage information about collidable points
    of a robot model, such as their positions and the corresponding bodies (links)
    they are rigidly attached to.

    Attributes:
        point:
            An array of shape (N, 3) representing the displacement of collidable points
            w.r.t the origin of their parent body.
        body:
            An array of integers representing the indices of the bodies (links)
            associated to each collidable point.
    """

    body: Static[tuple[int, ...]] = dataclasses.field(default_factory=lambda: [])

    point: jtp.Matrix = dataclasses.field(default_factory=lambda: jnp.array([]))

    @staticmethod
    def build_from(model_description: ModelDescription) -> GroundContact:
        """
        Build a GroundContact object from a model description.

        Args:
            model_description: The model description to consider.

        Returns:
            The GroundContact object.
        """

        if len(model_description.collision_shapes) == 0:
            return GroundContact()

        # Get all the links so that we can take their updated index.
        links_dict = {link.name: link for link in model_description}

        # Get all the enabled collidable points of the model.
        collidable_points = model_description.all_enabled_collidable_points()

        # Extract the positions L_p_C of the collidable points w.r.t. the link frames
        # they are rigidly attached to.
        points = jnp.vstack([cp.position for cp in collidable_points])

        # Extract the indices of the links to which the collidable points are rigidly
        # attached to.
        link_index_of_points = [
            links_dict[cp.parent_link.name].index for cp in collidable_points
        ]

        # Build the GroundContact object.
        gc = GroundContact(point=points, body=tuple(link_index_of_points))  # noqa

        assert gc.point.shape[1] == 3
        assert gc.point.shape[0] == len(gc.body)

        return gc
