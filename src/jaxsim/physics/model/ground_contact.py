import dataclasses

import jax.numpy as jnp
import jax_dataclasses
import numpy as np
import numpy.typing as npt
from jax_dataclasses import Static

from jaxsim.parsers.descriptions import ModelDescription


@jax_dataclasses.pytree_dataclass
class GroundContact:
    """
    A class for managing collidable points in a robot model.

    This class is used to store and manage information about collidable points on a robot model,
    such as their positions and the corresponding bodies (links) they are associated with.

    Attributes:
        point (npt.NDArray): An array of shape (3, N) representing the 3D positions of collidable points.
        body (Static[npt.NDArray]): An array of integers representing the indices of the bodies (links) associated with each collidable point.
    """

    point: npt.NDArray = dataclasses.field(default_factory=lambda: jnp.array([]))
    body: Static[npt.NDArray] = dataclasses.field(
        default_factory=lambda: np.array([], dtype=int)
    )

    @staticmethod
    def build_from(
        model_description: ModelDescription,
    ) -> "GroundContact":
        if len(model_description.collision_shapes) == 0:
            return GroundContact()

        # Get all the links so that we can take their updated index
        links_dict = {link.name: link for link in model_description}

        # Get all the enabled collidable points of the model
        collidable_points = model_description.all_enabled_collidable_points()

        # Build the GroundContact attributes
        points = jnp.vstack([cp.position for cp in collidable_points]).T
        link_index_of_points = np.array(
            [links_dict[cp.parent_link.name].index for cp in collidable_points]
        )

        # Build the object
        gc = GroundContact(point=points, body=link_index_of_points)

        assert gc.point.shape[0] == 3
        assert gc.point.shape[1] == len(gc.body)

        return gc
