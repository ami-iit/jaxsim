import abc

import gymnasium.spaces
import jax

import jaxsim.typing as jtp


class Space(abc.ABC):
    """"""

    # TODO: add num for multiple samples? Or if multiple keys -> multiple samples?
    @abc.abstractmethod
    def sample(self, key: jax.random.PRNGKey) -> jtp.PyTree:
        """"""
        pass

    @abc.abstractmethod
    def contains(self, x: jtp.PyTree) -> bool:
        """"""
        pass

    # @abc.abstractmethod
    # def to_gymnasium(self) -> gymnasium.Space:
    #     """"""
    #     pass

    def __contains__(self, x: jtp.PyTree) -> bool:
        """"""
        return self.contains(x)
