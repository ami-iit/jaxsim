from __future__ import annotations

import dataclasses
import pathlib
import tempfile
import warnings
from collections.abc import Sequence
from typing import Any

import mujoco as mj
import numpy as np
import numpy.typing as npt
import rod.urdf.exporter
from lxml import etree as ET
from scipy.spatial.transform import Rotation


def load_rod_model(
    model_description: str | pathlib.Path | rod.Model,
    is_urdf: bool | None = None,
    model_name: str | None = None,
) -> rod.Model:
    """
    Loads a ROD model from a URDF/SDF file or a ROD model.

    Args:
        model_description: The URDF/SDF file or ROD model to load.
        is_urdf: Whether to force parsing the model description as a URDF file.
        model_name: The name of the model to load from the resource.

    Returns:
        rod.Model: The loaded ROD model.
    """

    # Parse the SDF resource.
    sdf_element = rod.Sdf.load(sdf=model_description, is_urdf=is_urdf)

    # Fail if the SDF resource has no model.
    if len(sdf_element.models()) == 0:
        raise RuntimeError("Failed to find any model in the model description")

    # Return the model if there is only one.
    if len(sdf_element.models()) == 1:
        if model_name is not None and sdf_element.models()[0].name != model_name:
            raise ValueError(f"Model '{model_name}' not found in the description")

        return sdf_element.models()[0]

    # Require users to specify the model name if there are multiple models.
    if model_name is None:
        msg = "The resource has multiple models. Please specify the model name."
        raise ValueError(msg)

    # Build a dictionary of models in the resource for easy access.
    models = {m.name: m for m in sdf_element.models()}

    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found in the resource")

    return models[model_name]


class RodModelToMjcf:
    """"""

    @staticmethod
    def assets_from_rod_model(
        rod_model: rod.Model,
    ) -> dict[str, bytes]:
        """
        Generates a dictionary of assets from a ROD model.

        Args:
            rod_model: The ROD model to extract the assets from.

        Returns:
            dict: A dictionary of assets.
        """

        import resolve_robotics_uri_py

        assets_files = dict()

        for link in rod_model.links():
            for visual in link.visuals():
                if visual.geometry.mesh and visual.geometry.mesh.uri:
                    assets_files[visual.geometry.mesh.uri] = (
                        resolve_robotics_uri_py.resolve_robotics_uri(
                            visual.geometry.mesh.uri
                        )
                    )

            for collision in link.collisions():
                if collision.geometry.mesh and collision.geometry.mesh.uri:
                    assets_files[collision.geometry.mesh.uri] = (
                        resolve_robotics_uri_py.resolve_robotics_uri(
                            collision.geometry.mesh.uri
                        )
                    )

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
        Adds a floating joint to a URDF string.

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
        rod_model: rod.Model,
        considered_joints: list[str] | None = None,
        plane_normal: tuple[float, float, float] = (0, 0, 1),
        heightmap: bool | None = None,
        heightmap_samples_xy: tuple[int, int] = (101, 101),
        cameras: (
            MujocoCamera
            | Sequence[MujocoCamera]
            | dict[str, str]
            | Sequence[dict[str, str]]
        ) = (),
    ) -> tuple[str, dict[str, Any]]:
        """
        Converts a ROD model to a Mujoco MJCF string.

        Args:
            rod_model: The ROD model to convert.
            considered_joints: The list of joint names to consider in the conversion.
            plane_normal: The normal vector of the plane.
            heightmap: Whether to generate a heightmap.
            heightmap_samples_xy: The number of points in the heightmap grid.
            cameras: The custom cameras to add to the scene.

        Returns:
            A tuple containing the MJCF string and the dictionary of assets.
        """

        # -------------------------------------
        # Convert the model description to URDF
        # -------------------------------------

        # Consider all joints if not specified otherwise.
        considered_joints = set(
            considered_joints
            if considered_joints is not None
            else [j.name for j in rod_model.joints() if j.type != "fixed"]
        )

        # If considered joints are passed, make sure that they are all part of the model.
        if considered_joints - {j.name for j in rod_model.joints()}:
            extra_joints = set(considered_joints) - {j.name for j in rod_model.joints()}

            msg = f"Couldn't find the following joints in the model: '{extra_joints}'"
            raise ValueError(msg)

        # Create a dictionary of joints for quick access.
        joints_dict = {j.name: j for j in rod_model.joints()}

        # Convert all the joints not considered to fixed joints.
        for joint_name in {j.name for j in rod_model.joints()} - considered_joints:
            joints_dict[joint_name].type = "fixed"

        # Convert the ROD model to URDF.
        urdf_string = rod.urdf.exporter.UrdfExporter(
            gazebo_preserve_fixed_joints=False, pretty=True
        ).to_urdf_string(
            sdf=rod.Sdf(model=rod_model, version="1.7"),
        )

        # -------------------------------------
        # Add a floating joint if floating-base
        # -------------------------------------

        if not rod_model.is_fixed_base():
            considered_joints |= {"world_to_base"}
            urdf_string = RodModelToMjcf.add_floating_joint(
                urdf_string=urdf_string,
                base_link_name=rod_model.get_canonical_link(),
                floating_joint_name="world_to_base",
            )

        # ---------------------------------------
        # Inject the <mujoco> element in the URDF
        # ---------------------------------------

        parser = ET.XMLParser(remove_blank_text=True)
        root = ET.fromstring(text=urdf_string.encode(), parser=parser)

        mujoco_element = (
            ET.SubElement(root, "mujoco")
            if len(root.findall("./mujoco")) == 0
            else root.find("./mujoco")
        )

        _ = ET.SubElement(
            mujoco_element,
            "compiler",
            balanceinertia="true",
            discardvisual="false",
        )

        urdf_string = ET.tostring(root, pretty_print=True).decode()

        # ------------------------------
        # Post-process all dummy visuals
        # ------------------------------

        parser = ET.XMLParser(remove_blank_text=True)
        root: ET._Element = ET.fromstring(text=urdf_string.encode(), parser=parser)

        # Give a tiny radius to all dummy spheres
        for geometry in root.findall(".//visual/geometry[sphere]"):
            radius = np.fromstring(
                geometry.find("./sphere").attrib["radius"], sep=" ", dtype=float
            )
            if np.allclose(radius, np.zeros(1)):
                geometry.find("./sphere").set("radius", "0.001")

        # Give a tiny volume to all dummy boxes
        for geometry in root.findall(".//visual/geometry[box]"):
            size = np.fromstring(
                geometry.find("./box").attrib["size"], sep=" ", dtype=float
            )
            if np.allclose(size, np.zeros(3)):
                geometry.find("./box").set("size", "0.001 0.001 0.001")

        urdf_string = ET.tostring(root, pretty_print=True).decode()

        # ------------------------
        # Convert the URDF to MJCF
        # ------------------------

        # Load the URDF model into Mujoco.
        assets = RodModelToMjcf.assets_from_rod_model(rod_model=rod_model)
        mj_model = mj.MjModel.from_xml_string(xml=urdf_string, assets=assets)

        # Get the joint names.
        mj_joint_names = {
            mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_JOINT, idx)
            for idx in range(mj_model.njnt)
        }

        # Check that the Mujoco model only has the considered joints.
        if mj_joint_names != considered_joints:
            extra1 = mj_joint_names - considered_joints
            extra2 = considered_joints - mj_joint_names
            extra_joints = extra1.union(extra2)
            msg = "The Mujoco model has the following extra/missing joints: '{}'"
            raise ValueError(msg.format(extra_joints))

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".xml", prefix=f"{rod_model.name}_"
        ) as mjcf_file:

            # Convert the in-memory Mujoco model to MJCF.
            mj.mj_saveLastXML(mjcf_file.name, mj_model)

            # Parse the MJCF string as XML (etree).
            # We need to post-process the file to include additional elements.
            parser = ET.XMLParser(remove_blank_text=True)
            tree = ET.parse(source=mjcf_file, parser=parser)

        # Get the root element.
        root: ET._Element = tree.getroot()

        # Find the <mujoco> element (might be the root itself).
        mujoco_element: ET._Element = next(iter(root.iter("mujoco")))

        # --------------
        # Add the motors
        # --------------

        if len(mujoco_element.findall(".//actuator")) > 0:
            raise RuntimeError("The model already has <actuator> elements.")

        # Add the actuator element.
        actuator_element = ET.SubElement(mujoco_element, "actuator")

        # Add a motor for each joint.
        for joint_element in mujoco_element.findall(".//joint"):
            assert (
                joint_element.attrib["name"] in considered_joints
            ), joint_element.attrib["name"]
            if joint_element.attrib.get("type", "hinge") in {"free", "ball"}:
                continue
            ET.SubElement(
                actuator_element,
                "motor",
                name=f"{joint_element.attrib['name']}_motor",
                joint=joint_element.attrib["name"],
                gear="1",
            )

        # ---------------------------------------------
        # Set full transparency of collision geometries
        # ---------------------------------------------

        parser = ET.XMLParser(remove_blank_text=True)

        # Get all the (optional) names of the URDF collision elements
        collision_names = {
            c.attrib["name"]
            for c in ET.fromstring(text=urdf_string.encode(), parser=parser).findall(
                ".//collision[geometry]"
            )
            if "name" in c.attrib
        }

        # Set alpha=0 to the color of all collision elements
        for geometry_element in mujoco_element.findall(".//geom[@rgba]"):
            if geometry_element.attrib.get("name") in collision_names:
                r, g, b, _ = geometry_element.attrib["rgba"].split(" ")
                geometry_element.set("rgba", f"{r} {g} {b} 0")

        # -----------------------
        # Create the scene assets
        # -----------------------

        asset_element = (
            ET.SubElement(mujoco_element, "asset")
            if len(mujoco_element.findall(".//asset")) == 0
            else mujoco_element.find(".//asset")
        )

        _ = ET.SubElement(
            asset_element,
            "texture",
            type="skybox",
            builtin="gradient",
            rgb1="0.3 0.5 0.7",
            rgb2="0 0 0",
            width="512",
            height="512",
        )

        _ = ET.SubElement(
            asset_element,
            "texture",
            name="plane_texture",
            type="2d",
            builtin="checker",
            rgb1="0.1 0.2 0.3",
            rgb2="0.2 0.3 0.4",
            width="512",
            height="512",
            mark="cross",
            markrgb=".8 .8 .8",
        )

        _ = ET.SubElement(
            asset_element,
            "material",
            name="plane_material",
            texture="plane_texture",
            reflectance="0.2",
            texrepeat="5 5",
            texuniform="true",
        )

        _ = (
            ET.SubElement(
                asset_element,
                "hfield",
                name="terrain",
                nrow=f"{int(heightmap_samples_xy[0])}",
                ncol=f"{int(heightmap_samples_xy[1])}",
                # The following 'size' is a placeholder, it is updated dynamically
                # when a hfield/heightmap is stored into MjData.
                size="1 1 1 1",
            )
            if heightmap
            else None
        )

        # ----------------------------------
        # Populate the scene with the assets
        # ----------------------------------

        worldbody_scene_element = ET.SubElement(mujoco_element, "worldbody")

        _ = ET.SubElement(
            worldbody_scene_element,
            "geom",
            name="floor",
            type="plane" if not heightmap else "hfield",
            size="0 0 0.05",
            material="plane_material",
            condim="3",
            contype="1",
            conaffinity="1",
            zaxis=" ".join(map(str, plane_normal)),
            **({"hfield": "terrain"} if heightmap else {}),
        )

        _ = ET.SubElement(
            worldbody_scene_element,
            "light",
            name="sun",
            mode="fixed",
            directional="true",
            castshadow="true",
            pos="0 0 10",
            dir="0 0 -1",
        )

        # -------------------------------------------------------
        # Add a camera following the CoM of the worldbody element
        # -------------------------------------------------------

        worldbody_element = None

        # Find the <worldbody> element of our model by searching the one that contains
        # all the considered joints. This is needed because there might be multiple
        # <worldbody> elements inside <mujoco>.
        for wb in mujoco_element.findall(".//worldbody"):
            if all(
                wb.find(f".//joint[@name='{j}']") is not None for j in considered_joints
            ):
                worldbody_element = wb
                break

        if worldbody_element is None:
            raise RuntimeError("Failed to find the <worldbody> element of the model")

        # Camera attached to the model
        # It can be manually copied from `python -m mujoco.viewer --mjcf=<URDF_PATH>`
        _ = ET.SubElement(
            worldbody_element,
            "camera",
            name="track",
            mode="trackcom",
            pos="1.930 -2.279 0.556",
            xyaxes="0.771 0.637 0.000 -0.116 0.140 0.983",
            fovy="60",
        )

        # Add user-defined camera.
        for camera in cameras if isinstance(cameras, Sequence) else [cameras]:

            mj_camera = (
                camera
                if isinstance(camera, MujocoCamera)
                else MujocoCamera.build(**camera)
            )

            _ = ET.SubElement(worldbody_element, "camera", mj_camera.asdict())

        # ------------------------------------------------
        # Add a light following the  CoM of the first link
        # ------------------------------------------------

        if not rod_model.is_fixed_base():

            # Light attached to the model
            _ = ET.SubElement(
                worldbody_element,
                "light",
                name="light_model",
                mode="targetbodycom",
                target=worldbody_element.find(".//body").attrib["name"],
                directional="false",
                castshadow="true",
                pos="1 0 5",
            )

        # --------------------------------
        # Return the resulting MJCF string
        # --------------------------------

        mjcf_string = ET.tostring(root, pretty_print=True).decode()
        return mjcf_string, assets


class UrdfToMjcf:
    @staticmethod
    def convert(
        urdf: str | pathlib.Path,
        considered_joints: list[str] | None = None,
        model_name: str | None = None,
        plane_normal: tuple[float, float, float] = (0, 0, 1),
        heightmap: bool | None = None,
        cameras: (
            MujocoCamera
            | Sequence[MujocoCamera]
            | dict[str, str]
            | Sequence[dict[str, str]]
        ) = (),
    ) -> tuple[str, dict[str, Any]]:
        """
        Converts a URDF file to a Mujoco MJCF string.

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

        # Get the ROD model.
        rod_model = load_rod_model(
            model_description=urdf,
            is_urdf=True,
            model_name=model_name,
        )

        # Convert the ROD model to MJCF.
        return RodModelToMjcf.convert(
            rod_model=rod_model,
            considered_joints=considered_joints,
            plane_normal=plane_normal,
            heightmap=heightmap,
            cameras=cameras,
        )


class SdfToMjcf:
    @staticmethod
    def convert(
        sdf: str | pathlib.Path,
        considered_joints: list[str] | None = None,
        model_name: str | None = None,
        plane_normal: tuple[float, float, float] = (0, 0, 1),
        heightmap: bool | None = None,
        cameras: (
            MujocoCamera
            | Sequence[MujocoCamera]
            | dict[str, str]
            | Sequence[dict[str, str]]
        ) = (),
    ) -> tuple[str, dict[str, Any]]:
        """
        Converts a SDF file to a Mujoco MJCF string.

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

        # Get the ROD model.
        rod_model = load_rod_model(
            model_description=sdf,
            is_urdf=False,
            model_name=model_name,
        )

        # Convert the ROD model to MJCF.
        return RodModelToMjcf.convert(
            rod_model=rod_model,
            considered_joints=considered_joints,
            plane_normal=plane_normal,
            heightmap=heightmap,
            cameras=cameras,
        )


@dataclasses.dataclass
class MujocoCamera:
    """
    Helper class storing parameters of a Mujoco camera.

    Refer to the official documentation for more details:
    https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera
    """

    mode: str = "fixed"

    target: str | None = None
    fovy: str = "45"
    pos: str = "0 0 0"

    quat: str | None = None
    axisangle: str | None = None
    xyaxes: str | None = None
    zaxis: str | None = None
    euler: str | None = None

    name: str | None = None

    @classmethod
    def build(cls, **kwargs) -> MujocoCamera:

        if not all(isinstance(value, str) for value in kwargs.values()):
            raise ValueError("Values must be strings")

        return cls(**kwargs)

    @staticmethod
    def build_from_target_view(
        camera_name: str,
        lookat: Sequence[float | int] | npt.NDArray = (0, 0, 0),
        distance: float | int | npt.NDArray = 3,
        azimut: float | int | npt.NDArray = 90,
        elevation: float | int | npt.NDArray = -45,
        fovy: float | int | npt.NDArray = 45,
        degrees: bool = True,
        **kwargs,
    ) -> MujocoCamera:
        """
        Create a custom camera that looks at a target point.

        Note:
            The choice of the parameters is easier if we imagine to consider a target
            frame `T` whose origin is located over the lookat point and having the same
            orientation of the world frame `W`. We also introduce a camera frame `C`
            whose origin is located over the lower-left corner of the image, and having
            the x-axis pointing right and the y-axis pointing up in image coordinates.
            The camera renders what it sees in the -z direction of frame `C`.

        Args:
            camera_name: The name of the camera.
            lookat: The target point to look at (origin of `T`).
            distance:
                The distance from the target point (displacement between the origins
                of `T` and `C`).
            azimut:
                The rotation around z of the camera. With an angle of 0, the camera
                would loot at the target point towards the positive x-axis of `T`.
            elevation:
                The rotation around the x-axis of the camera frame `C`. Note that if
                you want to lift the view angle, the elevation is negative.
            fovy: The field of view of the camera.
            degrees: Whether the angles are in degrees or radians.
            **kwargs: Additional camera parameters.

        Returns:
            The custom camera.
        """

        # Start from a frame whose origin is located over the lookat point.
        # We initialize a -90 degrees rotation around the z-axis because due to
        # the default camera coordinate system (x pointing right, y pointing up).
        W_H_C = np.eye(4)
        W_H_C[0:3, 3] = np.array(lookat)
        W_H_C[0:3, 0:3] = Rotation.from_euler(
            seq="ZX", angles=[-90, 90], degrees=True
        ).as_matrix()

        # Process the azimut.
        R_az = Rotation.from_euler(seq="Y", angles=azimut, degrees=degrees).as_matrix()
        W_H_C[0:3, 0:3] = W_H_C[0:3, 0:3] @ R_az

        # Process elevation.
        R_el = Rotation.from_euler(
            seq="X", angles=elevation, degrees=degrees
        ).as_matrix()
        W_H_C[0:3, 0:3] = W_H_C[0:3, 0:3] @ R_el

        # Process distance.
        tf_distance = np.eye(4)
        tf_distance[2, 3] = distance
        W_H_C = W_H_C @ tf_distance

        # Extract the position and the quaternion.
        p = W_H_C[0:3, 3]
        Q = Rotation.from_matrix(W_H_C[0:3, 0:3]).as_quat(scalar_first=True)

        return MujocoCamera.build(
            name=camera_name,
            mode="fixed",
            fovy=f"{fovy if degrees else np.rad2deg(fovy)}",
            pos=" ".join(p.astype(str).tolist()),
            quat=" ".join(Q.astype(str).tolist()),
            **kwargs,
        )

    def asdict(self) -> dict[str, str]:

        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}
