from __future__ import annotations

import dataclasses
from typing import Any

import jax_dataclasses

import jaxsim.typing as jtp
from jaxsim.terrain.terrain import FlatTerrain, Terrain

from .common import ContactModel, ContactsParams


@jax_dataclasses.pytree_dataclass
class RigidContactParams(ContactsParams):
    """Parameters of the rigid contacts model."""

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, value: object) -> bool:
        return super().__eq__(value)


@jax_dataclasses.pytree_dataclass
class RigidContacts(ContactModel):
    """Rigid contacts model."""

    parameters: RigidContactParams = dataclasses.field(
        default_factory=RigidContactParams
    )

    terrain: jax_dataclasses.Static[Terrain] = dataclasses.field(
        default_factory=FlatTerrain
    )

    def compute_contact_forces(
        self, position: jtp.Vector, velocity: jtp.Vector, **kwargs
    ) -> tuple[jtp.Vector, tuple[Any, ...]]:
        pass
