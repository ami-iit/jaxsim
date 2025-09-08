import contextlib
import pathlib
from collections.abc import Iterator, Sequence

import mediapy as media
import mujoco as mj
import mujoco.viewer
import numpy as np
import numpy.typing as npt


class MujocoVideoRecorder:
    """
    Video recorder for the MuJoCo passive viewer.
    """

    def __init__(
        self,
        model: list[mj.MjModel] | mj.MjModel,
        data: list[mj.MjData] | mj.MjData,
        fps: int = 30,
        width: int | None = None,
        height: int | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the Mujoco video recorder.

        Args:
            model: The Mujoco model.
            data: The Mujoco data.
            fps: The frames per second.
            width: The width of the video.
            height: The height of the video.
            **kwargs: Additional arguments for the renderer.
        """

        if isinstance(model, mj.MjModel):
            single_model = model
        elif isinstance(model, list) and len(model) == 1:
            single_model = model[0]
        else:
            raise ValueError(
                "Model must be a single instance of mj.MjModel or a list with at least one element."
            )

        width = width if width is not None else single_model.vis.global_.offwidth
        height = height if height is not None else single_model.vis.global_.offheight

        if single_model.vis.global_.offwidth != width:
            single_model.vis.global_.offwidth = width

        if single_model.vis.global_.offheight != height:
            single_model.vis.global_.offheight = height

        self.fps = fps
        self.frames: list[npt.NDArray] = []
        self.data: list[mj.MjData] | mj.MjData | None = None
        self.model: list[mj.MjModel] | mj.MjModel | None = None
        self.reset(model=model, data=data)

        self.renderer = mujoco.Renderer(
            model=single_model,
            **(dict(width=width, height=height) | kwargs),
        )

    def reset(
        self,
        model: mj.MjModel | None = None,
        data: list[mj.MjData] | mj.MjData | None = None,
    ) -> None:
        """Reset the model and data."""

        self.frames = []

        self.data = data if data is not None else self.data
        self.data = self.data if isinstance(self.data, list) else [self.data]

        self.model = model if model is not None else self.model
        self.model = self.model if isinstance(self.model, list) else [self.model]

        assert len(self.data) == len(self.model) or len(self.model) == 1, (
            f"Length mismatch: len(data)={len(self.data)}, len(model)={len(self.model)}. "
            "They must be equal or model must have length 1."
        )

    def render_frame(self, camera_name: str = "track") -> npt.NDArray:
        """Render a frame."""

        for idx, data in enumerate(self.data):

            # Use a single model for rendering if multiple data instances are provided.
            # Otherwise, use the data index to select the corresponding model.
            model = self.model[0] if len(self.model) == 1 else self.model[idx]

            mj.mj_forward(model, data)

            if idx == 0:
                self.renderer.update_scene(data=data, camera=camera_name)
                continue

            mujoco.mjv_addGeoms(
                m=model,
                d=data,
                opt=mj.MjvOption(),
                pert=mj.MjvPerturb(),
                catmask=mj.mjtCatBit.mjCAT_DYNAMIC,
                scn=self.renderer.scene,
            )

        return self.renderer.render()

    def record_frame(self, camera_name: str = "track") -> None:
        """Store a frame in the buffer."""

        frame = self.render_frame(camera_name=camera_name)
        self.frames.append(frame)

    def write_video(self, path: pathlib.Path | str, exist_ok: bool = False) -> None:
        """Write the video to a file."""

        # Resolve the path to the video.
        path = pathlib.Path(path).expanduser().resolve()

        if path.is_dir():
            raise IsADirectoryError(f"The path '{path}' is a directory.")

        if not exist_ok and path.is_file():
            raise FileExistsError(f"The file '{path}' already exists.")

        media.write_video(path=path, images=np.array(self.frames), fps=self.fps)

    @staticmethod
    def compute_down_sampling(original_fps: int, target_min_fps: int) -> int:
        """
        Return the integer down-sampling factor to reach at least the target fps.

        Args:
            original_fps: The original fps.
            target_min_fps: The target minimum fps.

        Returns:
            The down-sampling factor.
        """

        down_sampling = 1
        down_sampling_final = down_sampling

        while original_fps / (down_sampling + 1) >= target_min_fps:
            down_sampling = down_sampling + 1

            if int(original_fps / down_sampling) == original_fps / down_sampling:
                down_sampling_final = down_sampling

        return down_sampling_final


class MujocoVisualizer:
    """
    Visualizer for the MuJoCo passive viewer.
    """

    def __init__(
        self, model: mj.MjModel | None = None, data: mj.MjData | None = None
    ) -> None:
        """
        Initialize the Mujoco visualizer.

        Args:
            model: The Mujoco model.
            data: The Mujoco data.
        """

        self.data = data
        self.model = model

    def sync(
        self,
        viewer: mj.viewer.Handle,
        model: mj.MjModel | None = None,
        data: mj.MjData | None = None,
    ) -> None:
        """Update the viewer with the current model and data."""

        data = data if data is not None else self.data
        model = model if model is not None else self.model

        mj.mj_forward(model, data)
        viewer.sync()

    def open_viewer(
        self,
        model: mj.MjModel | None = None,
        data: mj.MjData | None = None,
        show_left_ui: bool = False,
    ) -> mj.viewer.Handle:
        """Open a viewer."""

        data = data if data is not None else self.data
        model = model if model is not None else self.model

        handle = mj.viewer.launch_passive(
            model, data, show_left_ui=show_left_ui, show_right_ui=False
        )

        return handle

    @contextlib.contextmanager
    def open(
        self,
        model: mj.MjModel | None = None,
        data: mj.MjData | None = None,
        *,
        show_left_ui: bool = False,
        close_on_exit: bool = True,
        lookat: Sequence[float | int] | npt.NDArray | None = None,
        distance: float | int | npt.NDArray | None = None,
        azimuth: float | int | npt.NDArray | None = None,
        elevation: float | int | npt.NDArray | None = None,
    ) -> Iterator[mj.viewer.Handle]:
        """
        Context manager to open the Mujoco passive viewer.

        Note:
            Refer to the Mujoco documentation for details of the camera options:
            https://mujoco.readthedocs.io/en/stable/XMLreference.html#visual-global
        """

        handle = self.open_viewer(model=model, data=data, show_left_ui=show_left_ui)

        handle = MujocoVisualizer.setup_viewer_camera(
            viewer=handle,
            lookat=lookat,
            distance=distance,
            azimuth=azimuth,
            elevation=elevation,
        )

        try:
            yield handle
        finally:
            _ = handle.close() if close_on_exit else None

    @staticmethod
    def setup_viewer_camera(
        viewer: mj.viewer.Handle,
        *,
        lookat: Sequence[float | int] | npt.NDArray | None,
        distance: float | int | npt.NDArray | None = None,
        azimuth: float | int | npt.NDArray | None = None,
        elevation: float | int | npt.NDArray | None = None,
    ) -> mj.viewer.Handle:
        """
        Configure the initial viewpoint of the Mujoco passive viewer.

        Note:
            Refer to the Mujoco documentation for details of the camera options:
            https://mujoco.readthedocs.io/en/stable/XMLreference.html#visual-global

        Returns:
            The viewer with configured camera.
        """

        if lookat is not None:

            lookat_array = np.array(lookat, dtype=float).squeeze()

            if lookat_array.size != 3:
                raise ValueError(lookat)

            viewer.cam.lookat = lookat_array

        if distance is not None:
            viewer.cam.distance = float(distance)

        if azimuth is not None:
            viewer.cam.azimuth = float(azimuth) % 360

        if elevation is not None:
            viewer.cam.elevation = float(elevation)

        return viewer
