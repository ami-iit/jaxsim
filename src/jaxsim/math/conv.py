import jax.numpy as jnp

import jaxsim.typing as jtp

from .skew import Skew


class Convert:
    @staticmethod
    def coordinates_tf(X: jtp.Matrix, p: jtp.Matrix) -> jtp.Matrix:
        X = X.squeeze()
        p = p.squeeze()

        # If p has shape (X,), transform it to column vector
        p = jnp.vstack(p) if len(p.shape) == 1 else p
        rows_p, cols_p = p.shape

        if rows_p != 3:
            raise ValueError(p.shape)

        R = X[0:3, 0:3]
        r = -Skew.vee(R.T @ X[0:3, 3:6])

        if cols_p > 1:
            r = jnp.tile(r, (1, cols_p))

        assert r.shape == p.shape, (r.shape, p.shape)

        xp = R @ (p - r)
        return jnp.vstack(xp)

    @staticmethod
    def velocities_threed(v_6d: jtp.Matrix, p: jtp.Matrix) -> jtp.Matrix:
        v = v_6d.squeeze()
        p = p.squeeze()

        # If the arrays have shape (X,), transform them to column vectors
        v = jnp.vstack(v) if len(v.shape) == 1 else v
        p = jnp.vstack(p) if len(p.shape) == 1 else p

        rows_v, cols_v = v.shape
        rows_p, cols_p = p.shape

        if cols_v == 1 and cols_p > 1:
            v = jnp.repeat(v, cols_p, axis=1)

        if rows_v == 6:
            vp = v[0:3, :] + jnp.cross(v[3:6, :], p, axis=0)
        else:
            raise ValueError(v.shape)

        return jnp.vstack(vp)

    @staticmethod
    def forces_sixd(f_3d: jtp.Matrix, p: jtp.Matrix) -> jtp.Matrix:
        f = f_3d.squeeze()
        p = p.squeeze()

        # If the arrays have shape (X,), transform them to column vectors
        fp = jnp.vstack(f) if len(f.shape) == 1 else f
        p = jnp.vstack(p) if len(p.shape) == 1 else p

        rows_p, cols_p = p.shape
        rows_fp, cols_fp = fp.shape

        # Number of columns must match
        if cols_p != cols_fp:
            raise ValueError(cols_p, cols_fp)

        if rows_fp == 3:
            f = jnp.vstack([fp, jnp.cross(p, fp, axis=0)])
        else:
            raise ValueError(fp.shape)

        return jnp.vstack(f)
