import contextlib
import dataclasses
import functools
import inspect
import os
from typing import Any, Callable, Generator, TypeVar

import jax
import jax.flatten_util
from typing_extensions import ParamSpec

from jaxsim import logging
from jaxsim.utils import tracing

from . import Mutability, Vmappable

_P = ParamSpec("_P")
_R = TypeVar("_R")


class jax_tf:
    """
    Class containing decorators applicable to methods of Vmappable objects.
    """

    # Environment variables that can be used to disable the transformations
    EnvVarOOP: str = "JAXSIM_OOP_DECORATORS"
    EnvVarJitOOP: str = "JAXSIM_OOP_DECORATORS_JIT"
    EnvVarVmapOOP: str = "JAXSIM_OOP_DECORATORS_VMAP"
    EnvVarCacheOOP: str = "JAXSIM_OOP_DECORATORS_CACHE"

    @staticmethod
    def method_ro(
        fn: Callable[_P, _R],
        jit: bool = True,
        static_argnames: tuple[str, ...] | list[str] = (),
        vmap: bool | None = None,
        vmap_in_axes: tuple[int, ...] | int | None = None,
        vmap_out_axes: tuple[int, ...] | int | None = None,
    ) -> Callable[_P, _R]:
        """
        Decorator for r/o methods of classes inheriting from Vmappable.
        """

        return jax_tf.method(
            fn=fn,
            read_only=True,
            validate=True,
            jit_enabled=jit,
            static_argnames=static_argnames,
            vmap_enabled=vmap,
            vmap_in_axes=vmap_in_axes,
            vmap_out_axes=vmap_out_axes,
        )

    @staticmethod
    def method_rw(
        fn: Callable[_P, _R],
        validate: bool = True,
        jit: bool = True,
        static_argnames: tuple[str, ...] | list[str] = (),
        vmap: bool | None = None,
        vmap_in_axes: tuple[int, ...] | int | None = None,
        vmap_out_axes: tuple[int, ...] | int | None = None,
    ) -> Callable[_P, _R]:
        """
        Decorator for r/w methods of classes inheriting from Vmappable.
        """

        return jax_tf.method(
            fn=fn,
            read_only=False,
            validate=validate,
            jit_enabled=jit,
            static_argnames=static_argnames,
            vmap_enabled=vmap,
            vmap_in_axes=vmap_in_axes,
            vmap_out_axes=vmap_out_axes,
        )

    @staticmethod
    def method(
        fn: Callable[_P, _R],
        read_only: bool = True,
        validate: bool = True,
        jit_enabled: bool = True,
        static_argnames: tuple[str, ...] | list[str] = (),
        vmap_enabled: bool | None = None,
        vmap_in_axes: tuple[int, ...] | int | None = None,
        vmap_out_axes: tuple[int, ...] | int | None = None,
    ):
        """
        Decorator for methods of classes inheriting from Vmappable.

        This decorator enables executing the methods on an object characterized by a
        desired mutability, that is selected considering the r/o and validation flags.
        It also allows to transform the method with the jit/vmap transformations.
        If the Vmappable object is vectorized, the method is automatically vmapped, and
        the in_axes are properly post-processed to simplify the combination with jit.

        Args:
            fn: The method to decorate.
            read_only: Whether the method operates on a read-only object.
            validate: Whether r/w methods should preserve the pytree structure.
            jit_enabled: Whether to apply the jit transformation.
            static_argnames: The names of the arguments that should be static.
            vmap_enabled: Whether to apply the vmap transformation.
            vmap_in_axes: The in_axes to use for the vmap transformation.
            vmap_out_axes: The out_axes to use for the vmap transformation.

        Returns:
            The decorated method.
        """

        @functools.wraps(fn)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs):
            """The wrapper function that is returned by this decorator."""

            # Methods of classes inheriting from Vmappable decorated by this wrapper
            # automatically support jit/vmap/mutability features when called standalone.
            # However, when objects are arguments of plain functions transformed with
            # jit/vmap, and decorated methods are called inside those functions, we need
            # to disable this decorator to avoid double wrapping and execution errors.
            # We do so by iterating over the arguments, and checking whether they are
            # being traced by JAX.
            for argument in list(args) + list(kwargs.values()):
                try:
                    argument_flat, _ = jax.flatten_util.ravel_pytree(argument)

                    if tracing(argument_flat):
                        return fn(*args, **kwargs)
                except:
                    continue

            # ===============================================================
            # Wrap fn so that jit/vmap/mutability transformations are applied
            # ===============================================================

            # Initialize the mutability of the instance over which the method is running.
            # * In r/o methods, this approach prevents any type of mutation.
            # * In r/w methods, this approach allows to catch early JIT recompilations
            #   caused by unwanted changes in the pytree structure.
            if read_only:
                mutability = Mutability.FROZEN
            else:
                mutability = (
                    Mutability.MUTABLE if validate else Mutability.MUTABLE_NO_VALIDATION
                )

            # Extract the class instance over which fn is called
            instance: Vmappable = args[0]
            assert isinstance(instance, Vmappable)

            # Save the original mutability
            original_mutability = instance._mutability()

            # Inspect the environment to detect whether to enforce disabling jit/vmap
            deco_on = jax_tf.env_var_on(jax_tf.EnvVarOOP)
            jit_enabled_env = jax_tf.env_var_on(jax_tf.EnvVarJitOOP) and deco_on
            vmap_enabled_env = jax_tf.env_var_on(jax_tf.EnvVarVmapOOP) and deco_on

            # Allow disabling the cache of jit-compiled functions.
            # It can be useful for debugging or testing purposes.
            wrap_fn = (
                jax_tf.wrap_fn
                if jax_tf.env_var_on(jax_tf.EnvVarCacheOOP) and deco_on
                else jax_tf.wrap_fn.__wrapped__
            )

            # Get the transformed function (possibly cached by functools.cache).
            # Note that all the arguments of the following methods, when hashed, should
            # uniquely identify the returned function so that a new function is built
            # when arguments change and either jit or vmap have to be called again.
            fn_db = wrap_fn(
                fn=fn,  # noqa
                mutability=mutability,
                jit=jit_enabled_env and jit_enabled,
                static_argnames=tuple(static_argnames),
                vmap=vmap_enabled_env
                and (
                    vmap_enabled is True
                    or (vmap_enabled is None and instance.vectorized)
                ),
                in_axes=vmap_in_axes,
                out_axes=vmap_out_axes,
            )

            # Call the transformed (mutable/jit/vmap) method
            out, obj = fn_db(*args, **kwargs)

            if read_only:
                # Restore the original mutability
                instance._set_mutability(mutability=original_mutability)

                return out

            # =================================================================
            # From here we assume that the wrapper is operating on a r/w method
            # =================================================================

            from jax_dataclasses._dataclasses import JDC_STATIC_MARKER

            # Select the right runtime mutability. The only difference here is when a r/w
            # method is called on a frozen object. In this case, we enable updating the
            # pytree data and preserve its structure only if validation is enabled.
            mutability_dict = {
                Mutability.MUTABLE_NO_VALIDATION: Mutability.MUTABLE_NO_VALIDATION,
                Mutability.MUTABLE: Mutability.MUTABLE,
                Mutability.FROZEN: (
                    Mutability.MUTABLE if validate else Mutability.MUTABLE_NO_VALIDATION
                ),
            }

            # We need to replace all the dynamic leafs of the original instance with those
            # computed by the functional transformation.
            # We do so by iterating over the fields of the jax_dataclasses and ignoring
            # all the fields that are marked as static.
            # Caveats: https://github.com/ami-iit/jaxsim/pull/48#issuecomment-1746635121.
            with instance.mutable_context(
                mutability=mutability_dict[instance._mutability()]
            ):
                for f in dataclasses.fields(instance):  # noqa
                    if (
                        hasattr(f, "type")
                        and hasattr(f.type, "__metadata__")
                        and JDC_STATIC_MARKER in f.type.__metadata__
                    ):
                        continue

                    try:
                        setattr(instance, f.name, getattr(obj, f.name))
                    except AssertionError as exc:
                        logging.debug(f"Old object:\n{getattr(instance, f.name)}")
                        logging.debug(f"New object:\n{getattr(obj, f.name)}")
                        raise RuntimeError(
                            f"Failed to update field '{f.name}'"
                        ) from exc

            return out

        return wrapper

    @staticmethod
    @functools.cache
    def wrap_fn(
        fn: Callable,
        mutability: Mutability,
        jit: bool,
        static_argnames: tuple[str, ...] | list[str],
        vmap: bool,
        in_axes: tuple[int, ...] | int | None,
        out_axes: tuple[int, ...] | int | None,
    ) -> Callable:
        """
        Transform a method with jit/vmap and execute it on an object characterized
        by the desired mutability.

        Note:
            The method should take the object (self) as first argument.

        Note:
            This returned transformed method is cached by considering the hash of all
            the arguments. It will re-apply jit/vmap transformations only if needed.

        Args:
            fn: The method to consider.
            mutability: The mutability of the object on which the method is called.
            jit: Whether to apply jit transformations.
            static_argnames: The names of the arguments that should be considered static.
            vmap: Whether to apply vmap transformations.
            in_axes: The axes along which to vmap input arguments.
            out_axes: The axes along which to vmap output arguments.

        Note:
            In order to simplify the application of vmap, we close the method arguments
            over all the non-mapped input arguments. Furthermore, for improving the
            compatibility with jit, we also close the vmap application over the static
            arguments.

        Returns:
            The transformed method operating on an object with the desired mutability.
            We maintain the same signature of the original method.
        """

        # Extract the signature of the function
        sig = inspect.signature(fn)

        # All static arguments must be actual arguments of fn
        for name in static_argnames:
            if name not in sig.parameters:
                raise ValueError(f"Static argument '{name}' not found in {fn}")

        # If in_axes is a tuple, its dimension should match the number of arguments
        if isinstance(in_axes, tuple) and len(in_axes) != len(sig.parameters):
            msg = "The length of 'in_axes' must match the number of arguments ({})"
            raise ValueError(msg.format(len(sig.parameters)))

        # Check that static arguments are not mapped with vmap.
        # This case would not work since static arguments are not traces and vmap need
        # to trace arguments in order to map them.
        if isinstance(in_axes, tuple):
            for mapped_axis, arg_name in zip(in_axes, sig.parameters.keys()):
                if mapped_axis is not None and arg_name in static_argnames:
                    raise ValueError(
                        f"Static argument '{arg_name}' cannot be mapped with vmap"
                    )

        def fn_tf_vmap(*args, function_to_vmap: Callable, **kwargs):
            """Wrapper applying the vmap transformation"""

            # Canonicalize the arguments so that all of them are kwargs
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Build a dictionary mapping all arguments to a mapped axis, even when
            # the None is passed (defaults to in_axes=0) or and int is passed (defaults
            # to in_axes=<int>).
            match in_axes:
                case None:
                    argname_to_mapped_axis = {name: 0 for name in bound.arguments}
                case tuple():
                    argname_to_mapped_axis = {
                        name: in_axes[i] for i, name in enumerate(bound.arguments)
                    }
                case int():
                    argname_to_mapped_axis = {name: in_axes for name in bound.arguments}
                case _:
                    raise ValueError(in_axes)

            # Build a dictionary (argument_name -> argument) for all mapped arguments.
            # Note that a mapped argument is an argument whose axis is not None and
            # is not a static jit argument.
            vmap_mapped_args = {
                arg: value
                for arg, value in bound.arguments.items()
                if argname_to_mapped_axis[arg] is not None
                and arg not in static_argnames
            }

            # Build a dictionary (argument_name -> argument) for all unmapped arguments
            vmap_unmapped_args = {
                arg: value
                for arg, value in bound.arguments.items()
                if arg not in vmap_mapped_args
            }

            # Disable mapping of non-vectorized default arguments
            for arg, value in argname_to_mapped_axis.items():
                if arg in vmap_mapped_args and value == sig.parameters[arg].default:
                    logging.debug(f"Disabling vmapping of default argument '{arg}'")
                    argname_to_mapped_axis[arg] = None

            # Close the function over the unmapped arguments of vmap
            fn_closed = lambda *mapped_args: function_to_vmap(
                **vmap_unmapped_args, **dict(zip(vmap_mapped_args.keys(), mapped_args))
            )

            # Create the in_axes tuple of only the mapped arguments
            in_axes_mapped = tuple(
                argname_to_mapped_axis[name] for name in vmap_mapped_args
            )

            # If all in_axes are the same, simplify in_axes tuple to be just an integer
            if len(set(in_axes_mapped)) == 1:
                in_axes_mapped = list(set(in_axes_mapped))[0]

            # If, instead, in_axes has different elements, we need to replace the mapped
            # axis of "self" with a pytree having as leafs the mapped axis.
            # This is because the vmap in_axes specification must be a tree prefix of
            # the corresponding value.
            if isinstance(in_axes_mapped, tuple) and "self" in vmap_mapped_args:
                argname_to_mapped_axis["self"] = jax.tree_util.tree_map(
                    lambda _: argname_to_mapped_axis["self"], vmap_mapped_args["self"]
                )
                in_axes_mapped = tuple(
                    argname_to_mapped_axis[name] for name in vmap_mapped_args
                )

            # Apply the vmap transformation and call the function passing only the
            # mapped arguments. The unmapped arguments have been closed over.
            # Note: we altered the "in_axes" tuple so that it does not have any
            #       None elements.
            # Note: if "in_axes_mapped" is a tuple, the following fails if we pass kwargs,
            #       we need to pass the unpacked args tuple instead.
            return jax.vmap(
                fn_closed,
                in_axes=in_axes_mapped,
                **dict(out_axes=out_axes) if out_axes is not None else {},
            )(*list(vmap_mapped_args.values()))

        def fn_tf_jit(*args, function_to_jit: Callable, **kwargs):
            """Wrapper applying the jit transformation"""

            # Canonicalize the arguments so that all of them are kwargs
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Apply the jit transformation and call the function passing all arguments
            # as keyword arguments
            return jax.jit(function_to_jit, static_argnames=static_argnames)(
                **bound.arguments
            )

        # First applied wrapper that executes fn in a mutable context
        fn_mutable = functools.partial(
            jax_tf.call_class_method_in_mutable_context,
            fn=fn,
            jit=jit,
            mutability=mutability,
        )

        # Second applied wrapper that transforms fn with vmap
        fn_vmap = (
            fn_mutable
            if not vmap
            else functools.partial(fn_tf_vmap, function_to_vmap=fn_mutable)
        )

        # Third applied wrapper that transforms fn with jit
        fn_jit_vmap = (
            fn_vmap
            if not jit
            else functools.partial(fn_tf_jit, function_to_jit=fn_vmap)
        )

        return fn_jit_vmap

    @staticmethod
    def call_class_method_in_mutable_context(
        *args, fn: Callable, jit: bool, mutability: Mutability, **kwargs
    ) -> tuple[Any, Vmappable]:
        """
        Wrapper to call a method on an object with the desired mutable context.

        Args:
            fn: The method to call.
            jit: Whether the method is being jit compiled or not.
            mutability: The desired mutability context.
            *args: The positional arguments to pass to the method (including self).
            **kwargs: The keyword arguments to pass to the method.

        Returns:
            A tuple containing the return value of the method and the object
            possibly updated by the method if it is in read-write.

        Note:
            This approach enables to jit-compile methods of a stateful object without
            leaking traces, therefore obtaining a jax-compatible OOP pattern.
        """

        # Log here whether the method is being jit compiled or not.
        # This log message does not get printed from compiled code, so here is the
        # most appropriate place to be sure that we log it correctly.
        if jit:
            logging.debug(msg=f"JIT compiling {fn}")

        # Canonicalize the arguments so that all of them are kwargs
        sig = inspect.signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Extract the class instance over which fn is called
        instance: Vmappable = bound.arguments["self"]

        # Select the right mutability. If the instance is mutable with validation
        # disabled, we override the input mutability so that we do not fail in case
        # of mismatched tree structure.
        mut = (
            Mutability.MUTABLE_NO_VALIDATION
            if instance._mutability() is Mutability.MUTABLE_NO_VALIDATION
            else mutability
        )

        # Call fn in a mutable context
        with instance.mutable_context(mutability=mut):
            # Methods could call other decorated methods. When it happens, the decorator
            # of the called method is invoked, that applies jit and vmap transformations.
            # This is not desired as it calls vmap inside an already vmapped method.
            # We work around this occurrence by disabling the jit/vmap decorators of all
            # methods called inside fn through a context manager.
            # Note that we already work around this in the beginning of the wrapper
            # function by detecting traced arguments, but the decorator works also
            # when jit=False and vmap=False, therefore only enforcing the mutability.
            with jax_tf.disabled_oop_decorators():
                out = fn(**bound.arguments)

        return out, instance

    @staticmethod
    def env_var_on(var_name: str, default_value: str = "1") -> bool:
        """
        Check whether an environment variable is set to a value that is considered on.

        Args:
            var_name: The name of the environment variable.
            default_value: The default variable value to consider if the variable has not
                been exported.

        Returns:
            True if the environment variable contains an on value, False otherwise.
        """

        on_values = {"1", "true", "on", "yes"}
        return os.environ.get(var_name, default_value).lower() in on_values

    @staticmethod
    @contextlib.contextmanager
    def disabled_oop_decorators() -> Generator[None, None, None]:
        """
        Context manager to disable the application of jax transformations performed by
        the decorators of this class.

        Note: when the transformations are disabled, the only logic still applied is
              the selection of the object mutability over which the method is running.
        """

        # Check whether the environment variable is part of the environment and
        # save its value. We restore the original value before exiting the context.
        env_cache = (
            None if jax_tf.EnvVarOOP not in os.environ else os.environ[jax_tf.EnvVarOOP]
        )

        # Disable both jit and vmap transformations
        os.environ[jax_tf.EnvVarOOP] = "0"

        try:
            # Execute the code in the context with disabled transformations
            yield

        finally:
            # Restore the original value of the environment variable or remove it if
            # it was not present before entering the context
            if env_cache is not None:
                os.environ[jax_tf.EnvVarOOP] = env_cache
            else:
                _ = os.environ.pop(jax_tf.EnvVarOOP)
