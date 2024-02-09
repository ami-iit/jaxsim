import contextlib
import pathlib
from typing import ContextManager

import mujoco as mj
import mujoco.viewer


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
