import jax.numpy as jnp
import jax_dataclasses
import numpy as np
import numpy.typing as npt

from jaxsim.parsers.descriptions import ModelDescription


@jax_dataclasses.pytree_dataclass
class GroundContact:
    point: npt.NDArray = jax_dataclasses.field(default_factory=lambda: jnp.array([]))
    body: npt.NDArray = jax_dataclasses.static_field(
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
