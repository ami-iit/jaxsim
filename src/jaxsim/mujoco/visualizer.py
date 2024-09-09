import contextlib
import pathlib
from collections.abc import Sequence

import mediapy as media
import mujoco as mj
import mujoco.viewer
import numpy as np
import numpy.typing as npt


class MujocoVideoRecorder:
    """"""

    def __init__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
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

        width = width if width is not None else model.vis.global_.offwidth
        height = height if height is not None else model.vis.global_.offheight

        if model.vis.global_.offwidth != width:
            model.vis.global_.offwidth = width

        if model.vis.global_.offheight != height:
            model.vis.global_.offheight = height

        self.fps = fps
        self.frames: list[npt.NDArray] = []
        self.data: mujoco.MjData | None = None
        self.model: mujoco.MjModel | None = None
        self.reset(model=model, data=data)

        self.renderer = mujoco.Renderer(
            model=self.model,
            **(dict(width=width, height=height) | kwargs),
        )

    def reset(
        self, model: mj.MjModel | None = None, data: mj.MjData | None = None
    ) -> None:
        """Reset the model and data."""

        self.frames = []

        self.data = data if data is not None else self.data
        self.model = model if model is not None else self.model

    def render_frame(self, camera_name: str = "track") -> npt.NDArray:
        """Renders a frame."""

        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(data=self.data, camera=camera_name)

        return self.renderer.render()

    def record_frame(self, camera_name: str = "track") -> None:
        """Stores a frame in the buffer."""

        frame = self.render_frame(camera_name=camera_name)
        self.frames.append(frame)

    def write_video(self, path: pathlib.Path, exist_ok: bool = False) -> None:
        """Writes the video to a file."""

        # Resolve the path to the video.
        path = path.expanduser().resolve()

        if path.is_dir():
            raise IsADirectoryError(f"The path '{path}' is a directory.")

        if not exist_ok and path.is_file():
            raise FileExistsError(f"The file '{path}' already exists.")

        media.write_video(path=path, images=self.frames, fps=self.fps)

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
    """"""

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
        viewer: mujoco.viewer.Handle,
        model: mj.MjModel | None = None,
        data: mj.MjData | None = None,
    ) -> None:
        """Updates the viewer with the current model and data."""

        data = data if data is not None else self.data
        model = model if model is not None else self.model

        mj.mj_forward(model, data)
        viewer.sync()

    def open_viewer(
        self, model: mj.MjModel | None = None, data: mj.MjData | None = None
    ) -> mj.viewer.Handle:
        """Opens a viewer."""

        data = data if data is not None else self.data
        model = model if model is not None else self.model

        handle = mj.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        )

        return handle

    @contextlib.contextmanager
    def open(
        self,
        model: mj.MjModel | None = None,
        data: mj.MjData | None = None,
        *,
        close_on_exit: bool = True,
        lookat: Sequence[float | int] | npt.NDArray | None = None,
        distance: float | int | npt.NDArray | None = None,
        azimut: float | int | npt.NDArray | None = None,
        elevation: float | int | npt.NDArray | None = None,
    ) -> contextlib.AbstractContextManager[mujoco.viewer.Handle]:
        """
        Context manager to open the Mujoco passive viewer.

        Note:
            Refer to the Mujoco documentation for details of the camera options:
            https://mujoco.readthedocs.io/en/stable/XMLreference.html#visual-global
        """

        handle = self.open_viewer(model=model, data=data)

        handle = MujocoVisualizer.setup_viewer_camera(
            viewer=handle,
            lookat=lookat,
            distance=distance,
            azimut=azimut,
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
        azimut: float | int | npt.NDArray | None = None,
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

        if azimut is not None:
            viewer.cam.azimuth = float(azimut) % 360

        if elevation is not None:
            viewer.cam.elevation = float(elevation)

        return viewer
