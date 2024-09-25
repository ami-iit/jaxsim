from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np
import rod
import trimesh

from jaxsim import logging


def parse_object_mapping_object(obj) -> trimesh.Trimesh:
    if isinstance(obj, trimesh.Trimesh):
        return obj
    elif isinstance(obj, dict):
        if "type" not in obj:
            raise ValueError("Object type not specified")
        if obj["type"] == "box":
            if "extents" not in obj:
                raise ValueError("Box extents not specified")
            return trimesh.creation.box(extents=obj["extents"])
        elif obj["type"] == "sphere":
            if "radius" not in obj:
                raise ValueError("Sphere radius not specified")
            return trimesh.creation.icosphere(subdivisions=4, radius=obj["radius"])
        else:
            raise ValueError(f"Invalid object type {obj['type']}")
    elif isinstance(obj, rod.builder.primitive_builder.PrimitiveBuilder):
        raise NotImplementedError("PrimitiveBuilder not implemented")
    else:
        raise ValueError("Invalid object type")


class MeshMappingMethod(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def extract_points(self, mesh: trimesh.Trimesh) -> np.ndarray:
        self.mesh = mesh
        pass

    def __call__(self, mesh: trimesh.Trimesh):
        return self.extract_points(mesh=mesh)

    @abstractmethod
    def __str__(self):
        return self.__class__.__name__


class VertexExtraction(MeshMappingMethod):
    def __init__(self):
        pass

    def extract_points(self, mesh: trimesh.Trimesh) -> np.ndarray:
        super().extract_points(mesh)
        return self.mesh.vertices

    def __str__(self):
        return "VertexExtraction"


class RandomSurfaceSampling(MeshMappingMethod):
    def __init__(self, n: int = -1):
        if n <= 0 or n > len(self.mesh.vertices):
            logging.warning(
                "Invalid number of points for random surface sampling. Defaulting to all vertices"
            )
            n = len(self.mesh.vertices)
        self.n = n

    def extract_points(self, mesh: trimesh.Trimesh) -> np.ndarray:
        super().extract_points(mesh)
        return self.mesh.sample(self.n)

    def __str__(self):
        return f"RandomSurfaceSampling(n={self.n})"


class UniformSurfaceSampling(MeshMappingMethod):
    def __init__(self, n: int = -1):
        if n <= 0 or n > len(self.mesh.vertices):
            logging.warning(
                "Invalid number of points for uniform surface sampling. Defaulting to all vertices"
            )
            n = len(self.mesh.vertices)
        self.n = n

    def extract_points(self, mesh: trimesh.Trimesh) -> np.ndarray:
        super().extract_points(mesh)
        return trimesh.sample.sample_surface_even(mesh=self.mesh, count=self.n)

    def __str__(self):
        return f"UniformSurfaceSampling(n={self.n})"


class AAP(MeshMappingMethod):
    def __init__(self, axis: str, operator: str, value: float):
        valid_methods = [">", "<", ">=", "<="]
        if operator not in valid_methods:
            raise ValueError(
                f"Invalid method {operator} for AAP. Valid methods are {valid_methods}"
            )

        match (operator):
            case ">":
                self.aap_operator = np.greater
            case "<":
                self.aap_operator = np.less
            case ">=":
                self.aap_operator = np.greater_equal
            case "<=":
                self.aap_operator = np.less_equal
            case _:
                raise ValueError(f"Invalid method {operator} for AAP")

        self.axis = axis
        self.value = value

    def extract_points(self, mesh: trimesh.Trimesh) -> np.ndarray:
        super().extract_points(mesh)
        if self.axis == "x":
            points = self.mesh.vertices[
                self.aap_operator(self.mesh.vertices[:, 0], self.value)
            ]
        elif self.axis == "y":
            points = self.mesh.vertices[
                self.aap_operator(self.mesh.vertices[:, 1], self.value)
            ]
        elif self.axis == "z":
            points = self.mesh.vertices[
                self.aap_operator(self.mesh.vertices[:, 2], self.value)
            ]
        else:
            raise ValueError("Invalid axis for axis-aligned plane")

        return points

    def __str__(self):
        return (
            f"AAP(axis={self.axis}, operator={self.aap_operator}, value={self.value})"
        )


class SelectPointsOverAxis(MeshMappingMethod):
    def __init__(self, axis: str, direction: str, n: int):
        valid_dirs = ["higher", "lower"]
        if direction not in valid_dirs:
            raise ValueError(f"Invalid direction. Valid directions are {valid_dirs}")
        self.axis = axis
        self.direction = direction
        self.n = n

    def extract_points(self, mesh: trimesh.Trimesh) -> np.ndarray:
        super().extract_points(mesh)
        arr = self.mesh.vertices

        index = 0 if self.axis == "x" else 1 if self.axis == "y" else 2
        # Sort the array in ascending order
        sorted_arr = arr[arr[:, index].argsort()]

        if self.direction == "lower":
            # Select first N points
            points = sorted_arr[: self.n]
        elif self.direction == "higher":
            # Select last N points
            points = sorted_arr[-self.n :]
        else:
            raise ValueError(
                f"Invalid direction {self.direction} for SelectPointsOverAxis method"
            )

        return points

    def __str__(self):
        return f"SelectPointsOverAxis(axis={self.axis}, direction={self.direction}, n={self.n})"


class ObjectMapping(MeshMappingMethod):
    def __init__(
        self, objs: Sequence[trimesh.Trimesh | dict], method: str = "subtract"
    ):
        valid_methods = ["subtract", "intersect"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method {method} for object mapping")
        self.method = method

        self.objs: List[trimesh.Trimesh] = []
        for obj in objs:
            self.objs.append(parse_object_mapping_object(obj))

    def extract_points(self, mesh: trimesh.Trimesh) -> np.ndarray:
        super().extract_points(mesh)
        if len(self.objs) == 0:
            return mesh.vertices
        if self.method == "subtract":
            for obj in self.objs:
                mesh = mesh.difference(obj)
        elif self.method == "intersect":
            for obj in self.objs:
                mesh = mesh.intersection(obj)
        else:
            raise ValueError(f"Invalid method {self.method} for object mapping")

        return mesh.vertices

    def __str__(self):
        return f"ObjectMapping(method={self.method}, objs={len(self.objs)})"
