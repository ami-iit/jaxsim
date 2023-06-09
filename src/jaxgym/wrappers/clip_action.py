# from typing import Generic
# from jaxgym.functional import ActionFuncWrapper
# from jaxgym.functional.func_wrapper import WrapperActType
# from gymnasium.experimental.functional import ActType
# import gymnasium as gym
# import numpy as np
#
#
# class ClipActionWrapper(
#     ActionFuncWrapper[WrapperActType],
#     Generic[WrapperActType],
# ):
#     """"""
#
#     def action(self, action: WrapperActType) -> ActType:
#         """"""
#
#         if self.action_space.contains(x=action):
#             return action
#
#         assert isinstance(self.action_space, gym.spaces.Box)
#
#         return np.clip(
#             action, a_min=self.action_space.low, a_max=self.action_space.high
#         )
