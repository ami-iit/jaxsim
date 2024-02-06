import abc
from typing import Callable, Dict, Tuple

import jaxsim.typing as jtp
from jaxsim.high_level.model import StepData

ConfigureCallbackSignature = Callable[["jaxsim.JaxSim"], "jaxsim.JaxSim"]
PreStepCallbackSignature = Callable[
    ["jaxsim.JaxSim"], Tuple["jaxsim.JaxSim", jtp.PyTree]
]
PostStepCallbackSignature = Callable[
    ["jaxsim.JaxSim", Dict[str, StepData]], Tuple["jaxsim.JaxSim", jtp.PyTree]
]


class SimulatorCallback(abc.ABC):
    """
    A base class for simulator callbacks.
    """

    pass


class ConfigureCallback(SimulatorCallback):
    """
    A callback class to define logic for configuring the simulator before taking the first step.
    """

    @property
    def configure_cb(self) -> ConfigureCallbackSignature:
        return lambda sim: self.configure(sim=sim)

    @abc.abstractmethod
    def configure(self, sim: "jaxsim.JaxSim") -> "jaxsim.JaxSim":
        pass


class PreStepCallback(SimulatorCallback):
    """
    A callback class for performing actions before each simulation step.
    """

    @property
    def pre_step_cb(self) -> PreStepCallbackSignature:
        return lambda sim: self.pre_step(sim=sim)

    @abc.abstractmethod
    def pre_step(self, sim: "jaxsim.JaxSim") -> Tuple["jaxsim.JaxSim", jtp.PyTree]:
        pass


class PostStepCallback(SimulatorCallback):
    """
    A callback class for performing actions after each simulation step.
    """

    @property
    def post_step_cb(self) -> PostStepCallbackSignature:
        return lambda sim, step_data: self.post_step(sim=sim, step_data=step_data)

    @abc.abstractmethod
    def post_step(
        self, sim: "jaxsim.JaxSim", step_data: Dict[str, StepData]
    ) -> Tuple["jaxsim.JaxSim", jtp.PyTree]:
        pass


class CallbackHandler(ConfigureCallback, PreStepCallback, PostStepCallback):
    """
    A class that handles callbacks for the simulator.

    Note:
        The are different simulation stages with associated callbacks:
        - `configure`: runs before the first step is taken.
        - `pre_step`: runs at each step before integrating the dynamics and advancing the time.
        - `post_step`: runs at each step after the integration of the dynamics.
    """

    pass
