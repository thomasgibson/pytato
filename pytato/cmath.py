from __future__ import annotations

__copyright__ = """
Copyright (C) 2020 Andreas Kloeckner
Copyright (C) 2020 Matt Wala
Copyright (C) 2020 Xiaoyu Wei
Copyright (C) 2021 Kaushik Kulkarni
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

# {{{ docs

__doc__ = """
.. currentmodule:: pytato

.. autofunction:: abs
.. autofunction:: sqrt
.. autofunction:: sin
.. autofunction:: cos
.. autofunction:: tan
.. autofunction:: arcsin
.. autofunction:: arccos
.. autofunction:: arctan
.. autofunction:: conj
.. autofunction:: arctan2
.. autofunction:: sinh
.. autofunction:: cosh
.. autofunction:: tanh
.. autofunction:: exp
.. autofunction:: log
.. autofunction:: log10
.. autofunction:: isnan
.. autofunction:: real
.. autofunction:: imag
"""

# }}}

import numpy as np
import pymbolic.primitives as prim
from typing import Tuple, Optional
from pytato.array import Array, ArrayOrScalar, IndexLambda, _dtype_any
from pytato.scalar_expr import SCALAR_CLASSES
from pymbolic import var


def _apply_elem_wise_func(inputs: Tuple[ArrayOrScalar],
                          func_name: str,
                          ret_dtype: Optional[_dtype_any] = None
                          ) -> ArrayOrScalar:
    if all(isinstance(x, SCALAR_CLASSES) for x in inputs):
        np_func = getattr(np, func_name)
        return np_func(*inputs)  # type: ignore

    if not inputs:
        raise ValueError("at least one argument must be present")

    shape = None

    sym_args = []
    bindings = {}
    for index, inp in enumerate(inputs):
        if isinstance(inp, Array):
            if inp.dtype.kind not in ["f", "c"]:
                raise ValueError("only floating-point or complex "
                        "arguments supported")

            if shape is None:
                shape = inp.shape
            elif inp.shape != shape:
                # FIXME: merge this logic with arithmetic, so that broadcasting
                # is implemented properly
                raise NotImplementedError("broadcasting in function application")

            if ret_dtype is None:
                ret_dtype = inp.dtype

            bindings[f"in_{index}"] = inp
            sym_args.append(
                    prim.Subscript(var(f"in_{index}"),
                        tuple(var(f"_{i}") for i in range(len(shape)))))
        else:
            sym_args.append(inp)

    assert shape is not None
    assert ret_dtype is not None

    return IndexLambda(
            prim.Call(var(f"pytato.c99.{func_name}"), tuple(sym_args)),
            shape, ret_dtype, bindings)


def abs(x: Array) -> ArrayOrScalar:
    if x.dtype.kind == "c":
        result_dtype = np.empty(0, dtype=x.dtype).real.dtype
    else:
        result_dtype = x.dtype

    return _apply_elem_wise_func((x,), "abs", ret_dtype=result_dtype)


def sqrt(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "sqrt")


def sin(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "sin")


def cos(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "cos")


def tan(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "tan")


def arcsin(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "asin")


def arccos(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "acos")


def arctan(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "atan")


def conj(x: Array) -> ArrayOrScalar:
    if x.dtype.kind != "c":
        return x
    return _apply_elem_wise_func((x,), "conj")


def arctan2(y: Array, x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((y, x), "atan2")  # type:ignore


def sinh(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "sinh")


def cosh(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "cosh")


def tanh(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "tanh")


def exp(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "exp")


def log(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "log")


def log10(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "log10")


def isnan(x: Array) -> ArrayOrScalar:
    return _apply_elem_wise_func((x,), "isnan", np.dtype(np.int32))


def real(x: Array) -> ArrayOrScalar:
    if x.dtype.kind == "c":
        result_dtype = np.empty(0, dtype=x.dtype).real.dtype
    else:
        return x
    return _apply_elem_wise_func((x,), "real", ret_dtype=result_dtype)


def imag(x: Array) -> ArrayOrScalar:
    if x.dtype.kind == "c":
        result_dtype = np.empty(0, dtype=x.dtype).real.dtype
    else:
        import pytato as pt
        return pt.zeros(x.shape, dtype=x.dtype)
    return _apply_elem_wise_func((x,), "imag", ret_dtype=result_dtype)

# vim: fdm=marker
