import pathlib
import tempfile
import warnings
from collections.abc import Sequence
from typing import Any

import sdformat
from lxml import etree as ET

from jaxsim import logging

from .utils import MujocoCamera

MujocoCameraType = (
    MujocoCamera | Sequence[MujocoCamera] | dict[str, str] | Sequence[dict[str, str]]
)


def load_sdformat_model(
    model_description: str | pathlib.Path | sdformat.Model,
    is_urdf: bool | None = None,
    model_name: str | None = None,
) -> sdformat.Model:
    """
    Load an SDFormat model from a URDF/SDF file or an SDFormat model.

    Args:
        model_description: The URDF/SDF file or SDFormat model to load.
        is_urdf: Whether to force parsing the model description as a URDF file.
        model_name: The name of the model to load from the resource.

    Returns:
        sdformat.Model: The loaded SDFormat model.
    """

    # If it's already an SDFormat model, return it directly
    if isinstance(model_description, sdformat.Model):
        return model_description

    # Parse the SDF resource.
    root = sdformat.Root()

    # Determine if it's a file path or string content
    if isinstance(model_description, (str, pathlib.Path)):
        model_path = pathlib.Path(model_description)
        if model_path.exists():
            # It's a file path
            try:
                if is_urdf:
                    # For URDF files, we need to load them as URDF
                    root.load_sdf_string(str(model_path.read_text()))
                else:
                    root.load(str(model_path))
            except sdformat.SDFErrorsException as e:
                raise RuntimeError(f"Failed to load model file: {e}") from e
        else:
            # It's string content
            try:
                root.load_sdf_string(str(model_description))
            except sdformat.SDFErrorsException as e:
                raise RuntimeError(f"Failed to parse model string: {e}") from e

    # Try to get the model
    model = root.model()
    if model is not None:
        if model_name is not None and model.name() != model_name:
            raise ValueError(f"Model '{model_name}' not found in the description")
        return model

    # Check if the model is in a world
    if root.world_count() > 0:
        world = root.world_by_index(0)
        if world.model_count() == 0:
            raise RuntimeError("Failed to find any model in the model description")

        if world.model_count() == 1:
            model = world.model_by_index(0)
            if model_name is not None and model.name() != model_name:
                raise ValueError(f"Model '{model_name}' not found in the description")
            return model

        # Multiple models in world
        if model_name is None:
            msg = "The resource has multiple models. Please specify the model name."
            raise ValueError(msg)

        # Build a dictionary of models in the world for easy access.
        models = {
            world.model_by_index(i).name(): world.model_by_index(i)
            for i in range(world.model_count())
        }

        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found in the resource")

        return models[model_name]

    raise RuntimeError("Failed to find any model in the model description")


class ModelToMjcf:
    """
    Class to convert a URDF/SDF file or an SDFormat model to a Mujoco MJCF string.
    """

    @staticmethod
    def convert(
        model: str | pathlib.Path | sdformat.Model,
        considered_joints: list[str] | None = None,
        plane_normal: tuple[float, float, float] = (0, 0, 1),
        heightmap: bool | None = None,
        heightmap_samples_xy: tuple[int, int] = (101, 101),
        cameras: MujocoCameraType = (),
    ) -> tuple[str, dict[str, Any]]:
        """
        Convert a model to a Mujoco MJCF string.

        Args:
            model: The URDF/SDF file or SDFormat model to convert.
            considered_joints: The list of joint names to consider in the conversion.
            plane_normal: The normal vector of the plane.
            heightmap: Whether to generate a heightmap.
            heightmap_samples_xy: The number of points in the heightmap grid.
            cameras: The custom cameras to add to the scene.

        Returns:
            A tuple containing the MJCF string and the dictionary of assets.
        """

        match model:
            case sdformat.Model():
                sdformat_model = model
            case str() | pathlib.Path():
                # Convert the JaxSim model to an SDFormat model.
                sdformat_model = load_sdformat_model(
                    model_description=model,
                    is_urdf=None,
                    model_name=None,
                )
            case _:
                raise TypeError(f"Unsupported type for 'model': {type(model)}")

        # Convert the SDFormat model to MJCF.
        return SdformatModelToMjcf.convert(
            sdformat_model=sdformat_model,
            considered_joints=considered_joints,
            plane_normal=plane_normal,
            heightmap=heightmap,
            heightmap_samples_xy=heightmap_samples_xy,
            cameras=cameras,
        )


class SdformatModelToMjcf:
    """
    Class to convert an SDFormat model to a Mujoco MJCF string.
    """

    @staticmethod
    def assets_from_sdformat_model(
        sdformat_model: sdformat.Model,
    ) -> dict[str, bytes]:
        """
        Generate a dictionary of assets from an SDFormat model.

        Args:
            sdformat_model: The SDFormat model to extract the assets from.

        Returns:
            dict: A dictionary of assets.
        """

        import resolve_robotics_uri_py

        assets_files = dict()

        # Iterate through all links in the SDFormat model
        for link_idx in range(sdformat_model.link_count()):
            link = sdformat_model.link_by_index(link_idx)

            # Process visual elements
            for visual_idx in range(link.visual_count()):
                visual = link.visual_by_index(visual_idx)
                geometry = visual.geometry()

                # Check if it's a mesh geometry
                if geometry.type() == sdformat.GeometryType.MESH:
                    mesh = geometry.mesh()
                    if mesh and mesh.uri():
                        uri = mesh.uri()
                        try:
                            assets_files[uri] = (
                                resolve_robotics_uri_py.resolve_robotics_uri(uri)
                            )
                        except Exception:
                            # If URI resolution fails, skip this asset
                            pass

            # Process collision elements
            for collision_idx in range(link.collision_count()):
                collision = link.collision_by_index(collision_idx)
                geometry = collision.geometry()

                # Check if it's a mesh geometry
                if geometry.type() == sdformat.GeometryType.MESH:
                    mesh = geometry.mesh()
                    if mesh and mesh.uri():
                        uri = mesh.uri()
                        try:
                            assets_files[uri] = (
                                resolve_robotics_uri_py.resolve_robotics_uri(uri)
                            )
                        except Exception:
                            # If URI resolution fails, skip this asset
                            pass

        assets = {
            asset_name: asset.read_bytes() for asset_name, asset in assets_files.items()
        }

        return assets

    @staticmethod
    def add_floating_joint(
        urdf_string: str,
        base_link_name: str,
        floating_joint_name: str = "world_to_base",
    ) -> str:
        """
        Add a floating joint to a URDF string.

        Args:
            urdf_string: The URDF string to modify.
            base_link_name: The name of the base link to attach the floating joint.
            floating_joint_name: The name of the floating joint to add.

        Returns:
            str: The modified URDF string.
        """

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".urdf") as urdf_file:

            # Write the URDF string to a temporary file and move current position
            # to the beginning.
            urdf_file.write(urdf_string)
            urdf_file.seek(0)

            # Parse the MJCF string as XML (etree).
            parser = ET.XMLParser(remove_blank_text=True)
            tree = ET.parse(source=urdf_file, parser=parser)

        root: ET._Element = tree.getroot()

        if root.find(f".//joint[@name='{floating_joint_name}']") is not None:
            msg = f"The URDF already has a floating joint '{floating_joint_name}'"
            warnings.warn(msg, stacklevel=2)
            return ET.tostring(root, pretty_print=True).decode()

        # Create the "world" link if it doesn't exist.
        if root.find(".//link[@name='world']") is None:
            _ = ET.SubElement(root, "link", name="world")

        # Create the floating joint.
        world_to_base = ET.SubElement(
            root, "joint", name=floating_joint_name, type="floating"
        )

        # Check that the base link exists.
        if root.find(f".//link[@name='{base_link_name}']") is None:
            raise ValueError(f"Link '{base_link_name}' not found in the URDF")

        # Attach the floating joint to the base link.
        ET.SubElement(world_to_base, "parent", link="world")
        ET.SubElement(world_to_base, "child", link=base_link_name)

        urdf_string = ET.tostring(root, pretty_print=True).decode()
        return urdf_string

    @staticmethod
    def convert(
        sdformat_model: sdformat.Model,
        considered_joints: list[str] | None = None,
        plane_normal: tuple[float, float, float] = (0, 0, 1),
        heightmap: bool | None = None,
        heightmap_samples_xy: tuple[int, int] = (101, 101),
        cameras: MujocoCameraType = (),
    ) -> tuple[str, dict[str, Any]]:
        """
        Convert an SDFormat model to a Mujoco MJCF string.

        Args:
            sdformat_model: The SDFormat model to convert.
            considered_joints: The list of joint names to consider in the conversion.
            plane_normal: The normal vector of the plane.
            heightmap: Whether to generate a heightmap.
            heightmap_samples_xy: The number of points in the heightmap grid.
            cameras: The custom cameras to add to the scene.

        Returns:
            A tuple containing the MJCF string and the dictionary of assets.
        """

        # The full migration from ROD to SDFormat for Mujoco conversion is complex
        # and requires significant changes to the conversion pipeline.
        # This would involve:
        # 1. Converting SDFormat models to SDF strings or URDF export
        # 2. Handling joint filtering with SDFormat API
        # 3. Managing floating base models
        # 4. Asset extraction from SDFormat models
        # 5. Custom frame handling
        # 6. Integration with Mujoco's SDF/URDF loaders
        raise NotImplementedError(
            "SDFormat to Mujoco conversion is not yet fully implemented. "
            "The migration from ROD to SDFormat for the Mujoco module requires "
            "a complete rewrite of the conversion pipeline. "
            "This functionality will be available in a future release. "
            f"Attempted to convert model: {sdformat_model.name()}"
        )


class UrdfToMjcf:
    """
    Class to convert a URDF file to a Mujoco MJCF string.
    """

    @staticmethod
    def convert(
        urdf: str | pathlib.Path,
        considered_joints: list[str] | None = None,
        model_name: str | None = None,
        plane_normal: tuple[float, float, float] = (0, 0, 1),
        heightmap: bool | None = None,
        cameras: MujocoCameraType = (),
    ) -> tuple[str, dict[str, Any]]:
        """
        Convert a URDF file to a Mujoco MJCF string.

        Args:
            urdf: The URDF file to convert.
            considered_joints: The list of joint names to consider in the conversion.
            model_name: The name of the model to convert.
            plane_normal: The normal vector of the plane.
            heightmap: Whether to generate a heightmap.
            cameras: The list of cameras to add to the scene.

        Returns:
            tuple: A tuple containing the MJCF string and the assets dictionary.
        """

        logging.warning("This method is deprecated. Use 'ModelToMjcf.convert' instead.")

        # Use the new ModelToMjcf converter with SDFormat backend
        return ModelToMjcf.convert(
            model_description=urdf,
            is_urdf=True,
            model_name=model_name,
            considered_joints=considered_joints,
            plane_normal=plane_normal,
            heightmap=heightmap,
            cameras=cameras,
        )


class SdfToMjcf:
    """
    Class to convert a SDF file to a Mujoco MJCF string.
    """

    @staticmethod
    def convert(
        sdf: str | pathlib.Path,
        considered_joints: list[str] | None = None,
        model_name: str | None = None,
        plane_normal: tuple[float, float, float] = (0, 0, 1),
        heightmap: bool | None = None,
        cameras: MujocoCameraType = (),
    ) -> tuple[str, dict[str, Any]]:
        """
        Convert a SDF file to a Mujoco MJCF string.

        Args:
            sdf: The SDF file to convert.
            considered_joints: The list of joint names to consider in the conversion.
            model_name: The name of the model to convert.
            plane_normal: The normal vector of the plane.
            heightmap: Whether to generate a heightmap.
            cameras: The list of cameras to add to the scene.

        Returns:
            tuple: A tuple containing the MJCF string and the assets dictionary.
        """

        logging.warning("This method is deprecated. Use 'ModelToMjcf.convert' instead.")

        # Use the new ModelToMjcf converter with SDFormat backend
        return ModelToMjcf.convert(
            model_description=sdf,
            is_urdf=False,
            model_name=model_name,
            considered_joints=considered_joints,
            plane_normal=plane_normal,
            heightmap=heightmap,
            cameras=cameras,
        )
