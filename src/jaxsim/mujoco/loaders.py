import pathlib
import tempfile
import warnings
from typing import Any, Callable

import mujoco as mj
import rod.urdf.exporter
from lxml import etree as ET


def load_rod_model(
    model_description: str | pathlib.Path | rod.Model,
    is_urdf: bool | None = None,
    model_name: str | None = None,
) -> rod.Model:
    """"""

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


def generate_hfield(heightmap: Callable, size: tuple[int, int] = (10, 10)) -> str:
    """"""

    import numpy as np

    # Generate the heightmap.
    heightmap = heightmap(size)

    # Check the heightmap dimensions.
    if heightmap.shape != size:
        raise ValueError(
            f"Heightmap dimensions {heightmap.shape} do not match the size {size}"
        )

    # Create the hfield file.
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".hfield") as hfield_file:

        # Write the header.
        hfield_file.write(f"{size[0]} {size[1]}\n")

        # Write the heightmap.
        for row in heightmap:
            hfield_file.write(" ".join(map(str, row)) + "\n")

        # Move the current position to the beginning.
        hfield_file.seek(0)

        return hfield_file.name


class RodModelToMjcf:
    """"""

    @staticmethod
    def assets_from_rod_model(
        rod_model: rod.Model,
    ) -> dict[str, bytes]:
        """"""

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
        """"""

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
            warnings.warn(msg)
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
        heightmap: pathlib.Path | Callable | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """"""

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
        if considered_joints - set([j.name for j in rod_model.joints()]):
            extra_joints = set(considered_joints) - set(
                [j.name for j in rod_model.joints()]
            )
            msg = f"Couldn't find the following joints in the model: '{extra_joints}'"
            raise ValueError(msg)

        # Create a dictionary of joints for quick access.
        joints_dict = {j.name: j for j in rod_model.joints()}

        # Convert all the joints not considered to fixed joints.
        for joint_name in set(j.name for j in rod_model.joints()) - considered_joints:
            joints_dict[joint_name].type = "fixed"

        # Convert the ROD model to URDF.
        urdf_string = rod.urdf.exporter.UrdfExporter.sdf_to_urdf_string(
            sdf=rod.Sdf(model=rod_model, version="1.7"),
            gazebo_preserve_fixed_joints=False,
            pretty=True,
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
        # print(urdf_string)
        # raise

        # ------------------------------
        # Post-process all dummy visuals
        # ------------------------------

        parser = ET.XMLParser(remove_blank_text=True)
        root: ET._Element = ET.fromstring(text=urdf_string.encode(), parser=parser)
        import numpy as np

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
        mj_model = mj.MjModel.from_xml_string(xml=urdf_string, assets=assets)  # noqa

        # Get the joint names.
        mj_joint_names = set(
            mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_JOINT, idx)
            for idx in range(mj_model.njnt)
        )

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
        mujoco_element: ET._Element = list(root.iter("mujoco"))[0]

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
                r, g, b, a = geometry_element.attrib["rgba"].split(" ")
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

        # ----------------------------------
        # Populate the scene with the assets
        # ----------------------------------

        worldbody_scene_element = ET.SubElement(mujoco_element, "worldbody")

        if heightmap:
            _ = ET.SubElement(
                worldbody_scene_element,
                "geom",
                name="floor",
                type="hfield",
                size="10 10 0.05",
                material="plane_material",
                condim="3",
                contype="1",
                conaffinity="1",
                zaxis=" ".join(map(str, plane_normal)),
                file=(
                    heightmap
                    if isinstance(heightmap, pathlib.Path)
                    else generate_hfield(heightmap)
                ),
            )
        else:
            _ = ET.SubElement(
                worldbody_scene_element,
                "geom",
                name="floor",
                type="plane",
                size="0 0 0.05",
                material="plane_material",
                condim="3",
                contype="1",
                conaffinity="1",
                zaxis=" ".join(map(str, plane_normal)),
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
        _ = ET.SubElement(
            worldbody_element,
            "camera",
            name="track",
            mode="trackcom",
            pos="1.930 -2.279 0.556",
            xyaxes="0.771 0.637 0.000 -0.116 0.140 0.983",
            fovy="60",
        )

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
        heightmap: str | Callable | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """"""

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
        )


class SdfToMjcf:
    @staticmethod
    def convert(
        sdf: str | pathlib.Path,
        considered_joints: list[str] | None = None,
        model_name: str | None = None,
        plane_normal: tuple[float, float, float] = (0, 0, 1),
        heightmap: str | Callable | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """"""

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
        )
