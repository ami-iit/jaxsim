import contextlib
import pathlib
from typing import ContextManager

import mediapy as media
import mujoco as mj
import mujoco.viewer
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
        """"""

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
        """"""

        self.frames = []

        self.data = data if data is not None else self.data
        self.model = model if model is not None else self.model

    def render_frame(self, camera_name: str | None = None) -> None:
        """"""

        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(data=self.data)  # TODO camera name

        self.frames.append(self.renderer.render())

    def write_video(self, path: pathlib.Path, exist_ok: bool = False) -> None:
        """"""

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
        """"""

        self.data = data
        self.model = model

    def sync(
        self,
        viewer: mujoco.viewer.Handle,
        model: mj.MjModel | None = None,
        data: mj.MjData | None = None,
    ) -> None:
        """"""

        data = data if data is not None else self.data
        model = model if model is not None else self.model

        mj.mj_forward(model, data)
        viewer.sync()

    def open_viewer(
        self, model: mj.MjModel | None = None, data: mj.MjData | None = None
    ) -> mj.viewer.Handle:
        """"""

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
        close_on_exit: bool = True,
    ) -> ContextManager[mujoco.viewer.Handle]:
        """"""

        handle = self.open_viewer(model=model, data=data)

        try:
            yield handle
        finally:
            handle.close() if close_on_exit else None
