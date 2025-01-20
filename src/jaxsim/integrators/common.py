from typing import TypeVar

import jaxsim.typing as jtp

# =============
# Generic types
# =============

Time = jtp.FloatLike
TimeStep = jtp.FloatLike
State = NextState = TypeVar("State")
StateDerivative = TypeVar("StateDerivative")
PyTreeType = TypeVar("PyTreeType", bound=jtp.PyTree)
