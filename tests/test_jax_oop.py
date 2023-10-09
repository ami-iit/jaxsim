import dataclasses
import io
from contextlib import redirect_stdout
from typing import Any, Type

import jax
import jax.numpy as jnp
import jax_dataclasses
import numpy as np
import pytest

from jaxsim.utils import Mutability, Vmappable, oop

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@jax_dataclasses.pytree_dataclass
class AlgoData(Vmappable):
    """Class storing vmappable data of a given algorithm."""

    counter: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.array(0, dtype=jnp.uint64)
    )

    @classmethod
    def build(cls: Type[Self], counter: jax.typing.ArrayLike) -> Self:
        """Builder method. Helpful for enforcing type and shape of fields."""

        # Counter can be int / scalar numpy array / scalar jax array / etc.
        if jnp.array(counter).squeeze().size != 1:
            raise ValueError("The counter must be a scalar")

        # Create the object enforcing `counter` to be a scalar jax array
        data = AlgoData(
            counter=jnp.array(counter, dtype=jnp.uint64).squeeze(),
        )

        return data


def test_data():
    """Test AlgoData class."""

    data1 = AlgoData.build(counter=0)
    data2 = AlgoData.build(counter=np.array(10))
    data3 = AlgoData.build(counter=jnp.array(50))

    assert isinstance(data1.counter, jax.Array) and data1.counter.dtype == jnp.uint64
    assert isinstance(data2.counter, jax.Array) and data2.counter.dtype == jnp.uint64
    assert isinstance(data3.counter, jax.Array) and data3.counter.dtype == jnp.uint64

    assert data1.batch_size == 0
    assert data2.batch_size == 0
    assert data3.batch_size == 0

    # ==================
    # Vectorizing PyTree
    # ==================

    for batch_size in (0, 10, 100):
        data_vec = data1.vectorize(batch_size=batch_size)

        assert data_vec.batch_size == batch_size

        if batch_size > 0:
            assert data_vec.counter.shape[0] == batch_size

    # =========================================
    # Extracting element from vectorized PyTree
    # =========================================

    data_vec = AlgoData.build_from_list(list_of_obj=[data1, data2, data3])
    assert data_vec.batch_size == 3
    assert data_vec.extract_element(index=0) == data1
    assert data_vec.extract_element(index=1) == data2
    assert data_vec.extract_element(index=2) == data3

    with pytest.raises(ValueError):
        _ = data_vec.extract_element(index=3)

    out = data1.extract_element(index=0)
    assert out == data1
    assert id(out) != id(data1)

    with pytest.raises(RuntimeError):
        _ = data1.extract_element(index=1)

    with pytest.raises(ValueError):
        _ = AlgoData.build_from_list(list_of_obj=[data1, data2, data3, 42])


@jax_dataclasses.pytree_dataclass
class MyClassWithAlgorithms(Vmappable):
    """
    Class to demonstrate how to use `Vmappable`.
    """

    # Dynamic data of the algorithm
    data: AlgoData = dataclasses.field(default=None)

    # Static attribute of the pytree (triggers recompilation if changed)
    double_input: jax_dataclasses.Static[bool] = dataclasses.field(default=None)

    # Non-static attribute of the pytree that is not transparently vmap-able.
    done: jax.typing.ArrayLike = dataclasses.field(
        default_factory=lambda: jnp.array(False, dtype=bool)
    )

    # Additional leaves to test the behaviour of mutable and immutable python objects
    my_tuple: tuple[int] = dataclasses.field(default=tuple(jnp.array([1, 2, 3])))
    my_list: list[int] = dataclasses.field(
        default_factory=lambda: [4, 5, 6], init=False
    )
    my_array: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.array([10, 20, 30])
    )

    @classmethod
    def build(cls: Type[Self], double_input: bool = False) -> Self:
        """"""

        obj = MyClassWithAlgorithms()

        with obj.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
            obj.data = AlgoData.build(counter=0)
            obj.double_input = jnp.array(double_input)

        return obj

    @oop.jax_tf.method_ro
    def algo_ro(self, advance: int | jax.typing.ArrayLike) -> Any:
        """This is a read-only algorithm. It does not alter any pytree leaf."""

        # This should be printed only the first execution since it is disabled
        # in the execution of the JIT-compiled function.
        print("__algo_ro__")

        # Use the dynamic condition that doubles the input value
        mul = jax.lax.select(self.double_input, 2, 1)

        # Increase the counter
        counter_old = jnp.atleast_1d(self.data.counter)[0]
        counter_new = counter_old + mul * advance

        # Return the updated counter
        return counter_new

    @oop.jax_tf.method_rw
    def algo_rw(self, advance: int | jax.typing.ArrayLike) -> Any:
        """
        This is a read-write algorithm. It may alter pytree leaves either belonging
        to the vmappable data or generic non-static dataclass attributes.
        """

        print(self)

        # This should be printed only the first execution since it is disabled
        # in the execution of the JIT-compiled function.
        print("__algo_rw__")

        # Use the dynamic condition that doubles the input value
        mul = jax.lax.select(self.double_input, 2, 1)

        # Increase the internal counter
        counter_old = jnp.atleast_1d(self.data.counter)[0]
        self.data.counter = jnp.array(counter_old + mul * advance, dtype=jnp.uint64)

        # Update the non-static and non-vmap-able attribute
        self.done = jax.lax.cond(
            pred=self.data.counter > 100,
            true_fun=lambda _: jnp.array(True),
            false_fun=lambda _: jnp.array(False),
            operand=None,
        )

        print(self)

        # Return the updated counter
        return self.data.counter


def test_mutability():
    """Test MyClassWithAlgorithms class."""

    # Build the object
    obj_ro = MyClassWithAlgorithms.build(double_input=True)

    # By default, pytrees built with jax_dataclasses are frozen (read-only)
    assert obj_ro._mutability() == Mutability.FROZEN
    with pytest.raises(dataclasses.FrozenInstanceError):
        obj_ro.data.counter = 42

    # Data can be changed through a context manager, in this case operating on a copy...
    with obj_ro.editable(validate=True) as obj_ro_copy:
        obj_ro_copy.data.counter = jnp.array(42, dtype=obj_ro.data.counter.dtype)
    assert obj_ro_copy.data.counter == pytest.approx(42)
    assert obj_ro.data.counter != pytest.approx(42)

    # ... or a context manager that does not copy the pytree...
    with obj_ro.mutable_context(mutability=Mutability.MUTABLE):
        obj_ro.data.counter = jnp.array(42, dtype=obj_ro.data.counter.dtype)
    assert obj_ro.data.counter == pytest.approx(42)

    # ... that raises if the leafs change type
    with pytest.raises(AssertionError):
        with obj_ro.mutable_context(mutability=Mutability.MUTABLE):
            obj_ro.data.counter = 42

    # Pytrees can be copied...
    obj_ro_copy = obj_ro.copy()
    assert id(obj_ro) != id(obj_ro_copy)
    # ... operation that does not copy the leaves
    # TODO describe
    assert id(obj_ro.done) == id(obj_ro_copy.done)
    assert id(obj_ro.data.counter) == id(obj_ro_copy.data.counter)
    assert id(obj_ro.my_array) == id(obj_ro_copy.my_array)
    assert id(obj_ro.my_tuple) != id(obj_ro_copy.my_tuple)
    assert id(obj_ro.my_list) != id(obj_ro_copy.my_list)

    # They can be converted as mutable pytrees to update their values without
    # using context managers (maybe useful for debugging or quick prototyping)
    obj_rw = obj_ro.copy().mutable(validate=True)
    assert obj_rw._mutability() == Mutability.MUTABLE
    obj_rw.data.counter = jnp.array(42, dtype=obj_rw.data.counter.dtype)

    # However, with validation enabled, this works only if the leaf does not
    # change its type (shape, dtype, weakness, ...)
    with pytest.raises(AssertionError):
        obj_rw.data.counter = 100
    with pytest.raises(AssertionError):
        obj_rw.data.counter = jnp.array(100, dtype=float)
    with pytest.raises(AssertionError):
        obj_rw.data.counter = jnp.array([100, 200], dtype=obj_rw.data.counter.dtype)

    # Instead, with validation disabled, the pytree structure can be altered
    # (and this might cause JIT recompilations, so use it at your own risk)
    obj_rw_noval = obj_ro.copy().mutable(validate=False)
    assert obj_rw_noval._mutability() == Mutability.MUTABLE_NO_VALIDATION
    obj_rw_noval.data.counter = jnp.array(42, dtype=obj_rw.data.counter.dtype)

    # Now this should work without exceptions
    obj_rw_noval.data.counter = 100
    obj_rw_noval.data.counter = jnp.array(100, dtype=float)
    obj_rw_noval.data.counter = jnp.array([100, 200], dtype=obj_rw.data.counter.dtype)

    # Build another object and check mutability changes
    obj_ro = MyClassWithAlgorithms.build(double_input=True)
    assert obj_ro.is_mutable(validate=True) is False
    assert obj_ro.is_mutable(validate=False) is False

    obj_rw_val = obj_ro.mutable(validate=True)
    assert id(obj_ro) == id(obj_rw_val)
    assert obj_rw_val.is_mutable(validate=True) is True
    assert obj_rw_val.is_mutable(validate=False) is False

    obj_rw_noval = obj_rw_val.mutable(validate=False)
    assert id(obj_rw_noval) == id(obj_rw_val)
    assert obj_rw_noval.is_mutable(validate=True) is False
    assert obj_rw_noval.is_mutable(validate=False) is True

    # Checking mutable leaves behavior
    obj_rw = MyClassWithAlgorithms.build(double_input=True).mutable(validate=True)
    obj_rw_copy = obj_rw.copy()

    # Memory of JAX arrays cannot be altered in place so this is safe
    obj_rw.my_array = obj_rw.my_array.at[1].set(-20)
    assert obj_rw_copy.my_array[1] != -20

    # Tuples are immutable so this should be safe too
    obj_rw.my_tuple = tuple(jnp.array([1, -2, 3]))
    assert obj_rw_copy.my_array[1] != -2

    # Lists are treated as tuples (they are not leaves) but since they are mutable,
    # their id changes
    obj_rw.my_list[1] = -5
    assert obj_rw_copy.my_list[1] != -5

    # Check that exceptions in mutable context do not alter the object
    obj_ro = MyClassWithAlgorithms.build(double_input=True)
    assert obj_ro.data.counter == 0
    assert obj_ro.double_input == jnp.array(True)

    with pytest.raises(RuntimeError):
        with obj_ro.mutable_context(mutability=Mutability.MUTABLE):
            obj_ro.double_input = jnp.array(False, dtype=obj_ro.double_input.dtype)
            obj_ro.data.counter = jnp.array(33, dtype=obj_ro.data.counter.dtype)
            raise RuntimeError
    assert obj_ro.data.counter == 0
    assert obj_ro.double_input == jnp.array(True)


def test_decorators_jit_compilation():
    """Test JIT features of MyClassWithAlgorithms class."""

    obj = MyClassWithAlgorithms.build(double_input=False)
    assert obj.data.counter == 0
    assert obj.is_mutable(validate=True) is False
    assert obj.is_mutable(validate=False) is False

    # JIT compilation should happen only the first function call.
    # We test this by checking that the first execution prints some output.
    with io.StringIO() as buf, redirect_stdout(buf):
        _ = obj.algo_ro(advance=1)
        printed = buf.getvalue()
    assert "__algo_ro__" in printed
    with io.StringIO() as buf, redirect_stdout(buf):
        _ = obj.algo_ro(advance=1)
        printed = buf.getvalue()
    assert "__algo_ro__" not in printed

    # JIT compilation should happen only the first function call.
    # We test this by checking that the first execution prints some output.
    with io.StringIO() as buf, redirect_stdout(buf):
        _ = obj.algo_rw(advance=1)
        printed = buf.getvalue()
    assert "__algo_rw__" in printed
    with io.StringIO() as buf, redirect_stdout(buf):
        _ = obj.algo_rw(advance=1)
        printed = buf.getvalue()
    assert "__algo_rw__" not in printed

    # Create a new object
    obj = MyClassWithAlgorithms.build(double_input=False)

    # New objects should be able to re-use the JIT-compiled functions from other objects
    with io.StringIO() as buf, redirect_stdout(buf):
        _ = obj.algo_ro(advance=1)
        _ = obj.algo_rw(advance=1)
        printed = buf.getvalue()
    assert "__algo_ro__" not in printed
    assert "__algo_rw__" not in printed

    # Create a new object
    obj = MyClassWithAlgorithms.build(double_input=False)

    # Read-only methods can be called on r/o objects
    out = obj.algo_ro(advance=1)
    assert out == obj.data.counter + 1
    out = obj.algo_ro(advance=1)
    assert out == obj.data.counter + 1

    # Read-write methods can be called too on r/o objects since they are marked as r/w
    out = obj.algo_rw(advance=1)
    assert out == 1
    out = obj.algo_rw(advance=1)
    assert out == 2
    out = obj.algo_rw(advance=2)
    assert out == 4

    # Create a new object with a different dynamic attribute
    obj_dyn = MyClassWithAlgorithms.build(double_input=False).mutable(validate=True)
    obj_dyn.done = jnp.array(not obj_dyn.done, dtype=bool)

    # New objects with different dynamic attributes should be able to re-use the
    # JIT-compiled functions from other objects
    with io.StringIO() as buf, redirect_stdout(buf):
        _ = obj.algo_ro(advance=1)
        _ = obj.algo_rw(advance=1)
        printed = buf.getvalue()
    assert "__algo_ro__" not in printed
    assert "__algo_rw__" not in printed

    # Create a new object with a different static attribute
    obj_stat = MyClassWithAlgorithms.build(double_input=True)

    # New objects with different static attributes trigger the recompilation of the
    # JIT-compiled functions...
    with io.StringIO() as buf, redirect_stdout(buf):
        _ = obj_stat.algo_ro(advance=1)
        _ = obj_stat.algo_rw(advance=1)
        printed = buf.getvalue()
    assert "__algo_ro__" in printed
    assert "__algo_rw__" in printed

    # ... that are cached as well by jax
    with io.StringIO() as buf, redirect_stdout(buf):
        _ = obj_stat.algo_ro(advance=1)
        _ = obj_stat.algo_rw(advance=1)
        printed = buf.getvalue()
    assert "__algo_ro__" not in printed
    assert "__algo_rw__" not in printed


def test_decorators_vmap():
    """Test automatic vectorization features of MyClassWithAlgorithms class."""

    # Create a new object with scalar data
    obj = MyClassWithAlgorithms.build(double_input=False)

    # Vectorize the entire object
    obj_vec = obj.vectorize(batch_size=10)
    assert obj_vec.vectorized is True
    assert obj_vec.batch_size == 10
    assert id(obj_vec) != id(obj)

    # Calling methods of vectorized objects with scalar arguments should raise an error
    with pytest.raises(ValueError):
        _ = obj_vec.algo_ro(advance=1)
    with pytest.raises(ValueError):
        _ = obj_vec.algo_rw(advance=1)

    # Check that the r/o method provides automatically vectorized output and accepts
    # vectorized input
    out_vec = obj_vec.algo_ro(advance=jnp.array([1] * obj_vec.batch_size))
    assert out_vec.shape[0] == 10
    assert set(out_vec.tolist()) == {1}

    # Check that the r/w method provides automatically vectorized output and accepts
    # vectorized input
    out_vec = obj_vec.algo_rw(advance=jnp.array([1] * obj_vec.batch_size))
    assert out_vec.shape[0] == 10
    assert set(out_vec.tolist()) == {1}
    out_vec = obj_vec.algo_rw(advance=jnp.array([1] * obj_vec.batch_size))
    assert set(out_vec.tolist()) == {2}

    # Extract a single object from the vectorized object
    obj = obj_vec.extract_element(index=5)
    assert obj.vectorized is False
    assert obj.data.counter == obj_vec.data.counter[5]
