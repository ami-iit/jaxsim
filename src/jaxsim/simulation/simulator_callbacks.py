import abc
from typing import Callable, Dict, Tuple

import jaxsim.typing as jtp
from jaxsim.high_level.model import StepData
from jaxsim.simulation.simulator import JaxSim

ConfigureCallbackSignature = Callable[[JaxSim], JaxSim]
PreStepCallbackSignature = Callable[[JaxSim], JaxSim]
PostStepCallbackSignature = Callable[
    [JaxSim, Dict[str, StepData]], Tuple[JaxSim, jtp.PyTree]
]


class SimulatorCallback(abc.ABC):
    pass


class ConfigureCallback(SimulatorCallback):
    @property
    def configure_cb(self) -> ConfigureCallbackSignature:

        return lambda sim: self.configure(sim=sim)

    @abc.abstractmethod
    def configure(self, sim: JaxSim) -> JaxSim:
        pass


class PreStepCallback(SimulatorCallback):
    @property
    def pre_step_cb(self) -> PreStepCallbackSignature:

        return lambda sim: self.pre_step(sim=sim)

    @abc.abstractmethod
    def pre_step(self, sim: JaxSim) -> JaxSim:
        pass


class PostStepCallback(SimulatorCallback):
    @property
    def post_step_cb(self) -> PostStepCallbackSignature:

        return lambda sim, step_data: self.post_step(sim=sim, step_data=step_data)

    @abc.abstractmethod
    def post_step(
        self, sim: JaxSim, step_data: Dict[str, StepData]
    ) -> Tuple[JaxSim, jtp.PyTree]:
        pass


class CallbackHandler(ConfigureCallback, PreStepCallback, PostStepCallback):
    pass
