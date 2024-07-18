import trimesh
import numpy as np
import rod


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


class MeshMapping:
    @staticmethod
    def vertex_extraction(mesh: trimesh.Trimesh) -> np.ndarray:
        """Extracts the points of a mesh using the vertices of the mesh as colliders.

        Args:
            mesh: The mesh to extract the points from.

        Returns:
            The points of the mesh.
        """
        return mesh.vertices

    @staticmethod
    def random_surface_sampling(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
        """Extracts the points of a mesh by sampling the surface of the mesh randomly.

        Args:
            mesh: The mesh to extract the points from.
            num_points: The number of points to sample.

        Returns:
            The points of the mesh.
        """
        return mesh.sample(num_points)

    @staticmethod
    def uniform_surface_sampling(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
        """Extracts the points of a mesh by sampling the surface of the mesh uniformly.

        Args:
            mesh: The mesh to extract the points from.
            num_points: The number of points to sample.

        Returns:
            The points of the mesh.
        """
        return trimesh.sample.sample_surface_even(mesh=mesh, count=num_points)

    @staticmethod
    def aap(
        mesh: trimesh.Trimesh,
        axis: str,
        direction: str,
        aap_value: float,
    ) -> np.ndarray:
        """Axis Aligned Plane
        Extracts the points of a mesh that are on one side of an axis-aligned plane (AAP).
        This means that the algorithm considers all points above/below a certain value along one axis.

        Args:
            mesh: The mesh to extract the points from.
            axis: The axis of the AAP.
            direction: The direction of the AAP.
            aap_value: The value of the AAP.

        Returns:
            The points of the mesh that are on one side of the AAP.

        TODO: Implement inclined plane
        """
        if direction == "higher":
            aap_operator = np.greater
        elif direction == "lower":
            aap_operator = np.less
        else:
            raise ValueError("Invalid direction for axis-aligned plane")

        if axis == "x":
            points = mesh.vertices[aap_operator(mesh.vertices[:, 0], aap_value)]
        elif axis == "y":
            points = mesh.vertices[aap_operator(mesh.vertices[:, 1], aap_value)]
        elif axis == "z":
            points = mesh.vertices[aap_operator(mesh.vertices[:, 2], aap_value)]
        else:
            raise ValueError("Invalid axis for axis-aligned plane")

        return points

    @staticmethod
    def select_points_over_axis(
        mesh: trimesh.Trimesh,
        axis: str,
        direction: str,
        n: int,
    ):
        """Select Points Over Axis.
        Select N points over an axis, either starting from the lower or higher end.

        Args:
            mesh: The mesh to extract the points from.
            axis: The axis along which to remove points.
            direction: The direction of the AAP.
            n: The number of points to remove.

        Returns:
            The points of the mesh.
        """
        valid_dirs = ["higher", "lower"]
        if direction not in valid_dirs:
            raise ValueError(f"Invalid direction. Valid directions are {valid_dirs}")
        arr = mesh.vertices

        index = 0 if axis == "x" else 1 if axis == "y" else 2
        # Sort the array in ascending order
        sorted_arr = arr[arr[:, index].argsort()]

        if direction == "lower":
            # Select first N points
            points = sorted_arr[:n]
        elif direction == "higher":
            # Select last N points
            points = sorted_arr[-n:]
        else:
            raise ValueError(
                f"Invalid direction {direction} for SelectPointsOverAxis method"
            )

        return points

    @staticmethod
    def object_mapping(
        mesh: trimesh.Trimesh,
        object: trimesh.Trimesh,
        method: str = "subtraction",
        **kwargs,
    ):
        """Object Mapping.
        Removes points from a mesh that are inside another object, using subtraction or intersection.
        The method can be either "subtraction" or "intersection".
        The object can be a mesh or a primitive.

        Args:
            mesh: The mesh to extract the points from.
            object: The object to use for mapping.
            method: The method to use for mapping.
            **kwargs: Additional arguments for the method.

        Returns:
            The points of the mesh.
        """
        if method == "subtraction":
            x: trimesh.Trimesh = mesh.difference(object, **kwargs)
            x.show()
            points = x.vertices
        elif method == "intersection":
            points = mesh.intersection(object, **kwargs).vertices
        else:
            raise ValueError("Invalid method for object mapping")

        return points

    @staticmethod
    def mesh_decimation(
        mesh: trimesh.Trimesh,
        method: str = "",
        nsamples: int = -1,
    ):
        """Object decimation algorithm to reduce the number of vertices in a mesh, then extract points.

        Args:
            mesh: The mesh to extract the points from.
            method: The method to use for decimation.
            nsamples: The number of desired samples.

        Returns:
            The points of the mesh.
        """

        if method == "quadric":
            mesh = mesh.simplify_quadric_decimation(nsamples // 3)
        else:
            raise ValueError("Invalid method for mesh decimation")

        return mesh.vertices
