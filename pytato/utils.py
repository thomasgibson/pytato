from __future__ import annotations

__copyright__ = "Copyright (C) 2021 Kaushik Kulkarni"

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

import numpy as np
import islpy as isl
import pymbolic.primitives as prim

from typing import (Tuple, List, Union, Callable, Any, Sequence, Dict,
                    Optional, TypeVar, Iterable)
from pytato.array import (Array, ShapeType, IndexLambda, SizeParam, ShapeComponent,
                          DtypeOrScalar, ArrayOrScalar)
from pytato.scalar_expr import (ScalarExpression, IntegralScalarExpression,
                                SCALAR_CLASSES)
from pytools import UniqueNameGenerator
from pytato.transform import Mapper
from pytato.array import SliceItem


__doc__ = """
Helper routines
---------------

.. autofunction:: are_shape_components_equal
.. autofunction:: are_shapes_equal
.. autofunction:: get_shape_after_broadcasting
.. autofunction:: dim_to_index_lambda_components
"""


def get_shape_after_broadcasting(
        exprs: Sequence[Union[Array, ScalarExpression]]) -> ShapeType:
    """
    Returns the shape after broadcasting *exprs* in an operation.
    """
    shapes = [expr.shape if isinstance(expr, Array) else () for expr in exprs]

    result_dim = max((len(s) for s in shapes), default=0)

    # append leading dimensions of all the shapes with 1's to match result_dim.
    augmented_shapes = [((1,)*(result_dim-len(s)) + s) for s in shapes]

    def _get_result_axis_length(axis_lengths: List[IntegralScalarExpression]
                                ) -> IntegralScalarExpression:
        result_axis_len = axis_lengths[0]
        for axis_len in axis_lengths[1:]:
            if are_shape_components_equal(axis_len, result_axis_len):
                pass
            elif are_shape_components_equal(axis_len, 1):
                pass
            elif are_shape_components_equal(result_axis_len, 1):
                result_axis_len = axis_len
            else:
                raise ValueError("operands could not be broadcasted together with "
                                 f"shapes {' '.join(str(s) for s in shapes)}.")
        return result_axis_len

    return tuple(_get_result_axis_length([s[i] for s in augmented_shapes])
                 for i in range(result_dim))


def get_indexing_expression(shape: ShapeType,
                            result_shape: ShapeType) -> Tuple[ScalarExpression, ...]:
    """
    Returns the indices while broadcasting an array of shape *shape* into one of
    shape *result_shape*.
    """
    assert len(shape) <= len(result_shape)
    i_start = len(result_shape) - len(shape)
    indices = []
    for i, (dim1, dim2) in enumerate(zip(shape, result_shape[i_start:])):
        if not are_shape_components_equal(dim1, dim2):
            assert are_shape_components_equal(dim1, 1)
            indices.append(0)
        else:
            indices.append(prim.Variable(f"_{i+i_start}"))

    return tuple(indices)


def with_indices_for_broadcasted_shape(val: prim.Variable, shape: ShapeType,
                                       result_shape: ShapeType) -> prim.Expression:
    if len(shape) == 0:
        # scalar expr => do not index
        return val
    else:
        return val[get_indexing_expression(shape, result_shape)]


def extract_dtypes_or_scalars(
        exprs: Sequence[ArrayOrScalar]) -> List[DtypeOrScalar]:
    dtypes: List[DtypeOrScalar] = []
    for expr in exprs:
        if isinstance(expr, Array):
            dtypes.append(expr.dtype)
        else:
            assert isinstance(expr, SCALAR_CLASSES)
            dtypes.append(expr)

    return dtypes


def update_bindings_and_get_broadcasted_expr(arr: ArrayOrScalar,
                                             bnd_name: str,
                                             bindings: Dict[str, Array],
                                             result_shape: ShapeType
                                             ) -> ScalarExpression:
    """
    Returns an instance of :class:`~pytato.scalar_expr.ScalarExpression` to address
    *arr* in a :class:`pytato.array.IndexLambda` of shape *result_shape*.
    """

    if isinstance(arr, SCALAR_CLASSES):
        return arr

    assert isinstance(arr, Array)
    bindings[bnd_name] = arr
    return with_indices_for_broadcasted_shape(prim.Variable(bnd_name),
                                              arr.shape,
                                              result_shape)


def broadcast_binary_op(a1: ArrayOrScalar, a2: ArrayOrScalar,
                        op: Callable[[ScalarExpression, ScalarExpression], ScalarExpression],  # noqa:E501
                        get_result_type: Callable[[DtypeOrScalar, DtypeOrScalar], np.dtype[Any]],  # noqa:E501
                        ) -> ArrayOrScalar:
    if np.isscalar(a1) and np.isscalar(a2):
        from pytato.scalar_expr import evaluate
        return evaluate(op(a1, a2))  # type: ignore

    result_shape = get_shape_after_broadcasting([a1, a2])
    dtypes = extract_dtypes_or_scalars([a1, a2])
    result_dtype = get_result_type(*dtypes)

    bindings: Dict[str, Array] = {}

    expr1 = update_bindings_and_get_broadcasted_expr(a1, "_in0", bindings,
                                                     result_shape)
    expr2 = update_bindings_and_get_broadcasted_expr(a2, "_in1", bindings,
                                                     result_shape)

    return IndexLambda(op(expr1, expr2),
                       shape=result_shape,
                       dtype=result_dtype,
                       bindings=bindings)


# {{{ dim_to_index_lambda_components

class ShapeExpressionMapper(Mapper):
    """
    Mapper that takes a shape component and returns it as a scalar expression.
    """
    def __init__(self, var_name_gen: UniqueNameGenerator):
        self.cache: Dict[Array, ScalarExpression] = {}
        self.var_name_gen = var_name_gen
        self.bindings: Dict[str, SizeParam] = {}

    def rec(self, expr: Array) -> ScalarExpression:  # type: ignore
        if expr in self.cache:
            return self.cache[expr]
        result: Array = super().rec(expr)
        self.cache[expr] = result
        return result

    def map_index_lambda(self, expr: IndexLambda) -> ScalarExpression:
        from pytato.scalar_expr import substitute
        return substitute(expr.expr, {name: self.rec(val)
                                      for name, val in expr.bindings.items()})

    def map_size_param(self, expr: SizeParam) -> ScalarExpression:
        name = self.var_name_gen("_in")
        self.bindings[name] = expr
        return prim.Variable(name)


def dim_to_index_lambda_components(expr: ShapeComponent,
                                   vng: Optional[UniqueNameGenerator] = None,
                                   ) -> Tuple[ScalarExpression,
                                              Dict[str, SizeParam]]:
    """
    Returns the scalar expressions and bindings to use the shape
    component within an index lambda.

    .. testsetup::

        >>> import pytato as pt
        >>> from pytato.utils import dim_to_index_lambda_components
        >>> from pytools import UniqueNameGenerator

    .. doctest::

        >>> n = pt.make_size_param("n")
        >>> expr, bnds = dim_to_index_lambda_components(3*n+8, UniqueNameGenerator())
        >>> print(expr)
        3*_in + 8
        >>> bnds  # doctest: +ELLIPSIS
        {'_in': <pytato.array.SizeParam ...>}
    """
    if isinstance(expr, int):
        return expr, {}

    if vng is None:
        vng = UniqueNameGenerator()

    assert isinstance(vng, UniqueNameGenerator)
    assert isinstance(expr, Array)
    mapper = ShapeExpressionMapper(vng)
    result = mapper(expr)
    return result, mapper.bindings

# }}}


def are_shape_components_equal(dim1: ShapeComponent, dim2: ShapeComponent) -> bool:
    """
    Returns *True* iff *dim1* and *dim2* are have equal
    :class:`~pytato.array.SizeParam` coefficients in their expressions.
    """
    from pytato.scalar_expr import substitute, distribute

    def to_expr(dim: ShapeComponent) -> ScalarExpression:
        expr, bnds = dim_to_index_lambda_components(dim,
                                                    UniqueNameGenerator())

        return substitute(expr, {name: prim.Variable(bnd.name)
                                 for name, bnd in bnds.items()})

    dim1_expr = to_expr(dim1)
    dim2_expr = to_expr(dim2)
    # ScalarExpression.__eq__  returns Any
    return (distribute(dim1_expr-dim2_expr) == 0)  # type: ignore


def are_shapes_equal(shape1: ShapeType, shape2: ShapeType) -> bool:
    """
    Returns *True* iff *shape1* and *shape2* have the same dimensionality and the
    correpsonding components are equal as defined by
    :func:`~pytato.utils.are_shape_components_equal`.
    """
    return ((len(shape1) == len(shape2))
            and all(are_shape_components_equal(dim1, dim2)
                    for dim1, dim2 in zip(shape1, shape2)))


# {{{ ShapeToISLExpressionMapper

class ShapeToISLExpressionMapper(Mapper):
    """
    Mapper that takes a shape component and returns it as :class:`isl.Aff`.
    """
    def __init__(self, space: isl.Space):
        self.cache: Dict[Array, isl.Aff] = {}
        self.space = space

    def rec(self, expr: Array) -> isl.Aff:  # type: ignore
        if expr in self.cache:
            return self.cache[expr]
        result: Array = super().rec(expr)
        self.cache[expr] = result
        return result

    def map_index_lambda(self, expr: IndexLambda) -> isl.Aff:
        from pytato.scalar_expr import evaluate
        return evaluate(expr.expr, {name: self.rec(val)
                                    for name, val in expr.bindings.items()})

    def map_size_param(self, expr: SizeParam) -> isl.Aff:
        dt, pos = self.space.get_var_dict()[expr.name]
        return isl.Aff.var_on_domain(self.space, dt, pos)


# }}}


def _create_size_param_space(names: Iterable[str]) -> isl.Space:
    return isl.Space.create_from_names(isl.DEFAULT_CONTEXT,
                                       set=[],
                                       params=sorted(names)).params()


def _get_size_params_assumptions_bset(space: isl.Space) -> isl.BasicSet:
    bset = isl.BasicSet.universe(space)
    for name in bset.get_var_dict():
        bset = bset.add_constraint(isl.Constraint.ineq_from_names(space, {name: 1}))

    return bset


def _is_positive(expr: ShapeComponent) -> bool:
    """
    Returns *True* iff it can be proven that ``expr >= 0``.
    """
    if isinstance(expr, int):
        return expr >= 0

    assert isinstance(expr, Array)
    from pytato.transform import InputGatherer
    # type-ignore reason: passed Set[Optional[str]]; function expects Set[str]
    space = _create_size_param_space({expr.name  # type: ignore
                                      for expr in InputGatherer()(expr)})
    aff = ShapeToISLExpressionMapper(space)(expr)
    # type-ignore reason: mypy doesn't know comparing isl.Sets returns bool
    return (aff.ge_set(aff * 0)  # type: ignore
            <= _get_size_params_assumptions_bset(space))


def _is_negative(expr: ShapeComponent) -> bool:
    """
    Returns *True* iff it can be proven that ``expr <= 0``.
    """
    if isinstance(expr, int):
        return expr <= 0

    assert isinstance(expr, Array)
    from pytato.transform import InputGatherer
    # type-ignore reason: passed Set[Optional[str]]; function expects Set[str]
    space = _create_size_param_space({expr.name  # type: ignore
                                      for expr in InputGatherer()(expr)})
    aff = ShapeToISLExpressionMapper(space)(expr)
    # type-ignore reason: mypy doesn't know comparing isl.Sets returns bool
    return (aff.le_set(aff * 0)  # type: ignore
            <= _get_size_params_assumptions_bset(space))


# {{{ _index_into

Tfind = TypeVar("Tfind")


def _find(seq: Sequence[Tfind], key: Callable[[Tfind], bool]) -> int:
    i = 0
    for val in seq:
        if key(val):
            break
        i += 1
    return i


def _normalize_slice(slice_: slice,
                     axis_len: ShapeComponent) -> Tuple[ShapeComponent,
                                                        ShapeComponent,
                                                        int]:
    start, stop, step = slice_.start, slice_.stop, slice_.step
    if step is None:
        step = 1
    if not isinstance(step, int):
        raise ValueError(f"slice step must be an int or 'None' (got a {type(step)})")
    if step == 0:
        raise ValueError("slice step cannot be zero")

    if step > 0:
        _DEFAULT_START_: ShapeComponent = 0  # noqa: N806
        _DEFAULT_STOP_: ShapeComponent = axis_len  # noqa: N806
    else:
        _DEFAULT_START_ = axis_len - 1  # noqa: N806
        _DEFAULT_STOP_ = -1  # noqa: N806

    if start is None:
        start = _DEFAULT_START_
    else:
        if isinstance(axis_len, int):
            if -axis_len <= start < axis_len:
                start = start % axis_len
            else:
                start = 0
        else:
            raise NotImplementedError

    if stop is None:
        stop = _DEFAULT_STOP_
    else:
        if isinstance(axis_len, int):
            if -axis_len <= stop < axis_len:
                stop = stop % axis_len
            else:
                stop = axis_len
        else:
            raise NotImplementedError

    return start, stop, step


def _index_into(ary: Array, index: Tuple[SliceItem, ...]) -> IndexLambda:

    # {{{ handle ellipsis

    if index.count(...) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")

    if index.count(...):
        ellipsis_pos = index.index(...)
        index = (index[:ellipsis_pos]
                 + (slice(None, None, None),) * (ary.ndim - len(index) + 1)
                 + index[ellipsis_pos+1:])

    # }}}

    # {{{ "pad" index with complete slices to match ary's ndim

    if len(index) < ary.ndim:
        index = index + (slice(None, None, None),) * (ary.ndim - len(index))

    # }}}

    if len(index) != ary.ndim:
        raise IndexError(f"Too many indices (expected {ary.ndim}, got {len(index)})")

    if any(idx is None for idx in index):
        raise NotImplementedError("newaxis is not supported")

    arys = [idx for idx in index if isinstance(idx, Array)]

    try:
        indirection_shape = get_shape_after_broadcasting(arys)
    except ValueError as e:
        raise IndexError(str(e))

    # {{{ validate types

    for idx in index:
        if isinstance(idx, (slice, int)):
            pass
        elif isinstance(idx, Array):
            if idx.dtype.kind != "i":
                raise IndexError("only integer arrays are valid array indices")
        else:
            raise IndexError("only integers, slices, ellipsis and integer arrays"
                             " are valid indices")

    # }}}

    vng = UniqueNameGenerator()
    lhs_shape: List[ShapeComponent] = []
    rhs_indices: List[ScalarExpression] = []
    bindings = {}
    inserted_indirection_shape = False
    indirection_start = _find(index, lambda x: isinstance(x, Array))

    # {{{ avoid reading too much into numpy-spec for uncommon cases.

    # x = np.random.rand(3, 3, 3, 3)
    # idx1 = np.array([[-1], [1]])
    # idx2 = np.array([0, 2])
    # (a) x[idx1, idx2, :, :] has a shape of (2, 2, 3, 3)
    # (b) x[:, :, idx1, idx2] has a shape of (3, 3, 2, 2)
    # (c) x[:, idx1, idx2, :] has a shape of (3, 2, 2, 3)
    # (d) x[:, idx1, :, idx2] has a shape of (2, 2, 3, 3)
    # Case (d)'s shape seems completely arbitrary => do not try to reverse
    # engineer

    for i in range(indirection_start, indirection_start+len(arys)):
        if not isinstance(index[i], Array):
            raise NotImplementedError("Non-contiguous sequences of array "
                                      "indices is not properly defined in numpy"
                                      "-spec.")

    # }}}

    for i, idx in enumerate(index):
        axis_len = ary.shape[i]
        if isinstance(idx, int):
            if isinstance(axis_len, int):
                if -axis_len <= idx < axis_len:
                    rhs_indices.append(idx % axis_len)
                else:
                    raise IndexError(f"{idx} is out of bounds for axis {i}")
            else:
                raise NotImplementedError
        elif isinstance(idx, slice):
            start, stop, step = _normalize_slice(idx, ary.shape[i])
            rhs_indices.append(start + step * prim.Variable(f"_{len(lhs_shape)}"))

            if step > 0:
                if _is_positive(stop - start):
                    lhs_shape.append((stop - start) // step)
                elif _is_negative(stop - start):
                    lhs_shape.append(0)
                else:
                    raise NotImplementedError
            else:
                if _is_positive(start - stop):
                    lhs_shape.append((start - stop) // (-step))
                elif _is_negative(start - stop):
                    lhs_shape.append(0)
                else:
                    raise NotImplementedError
        elif isinstance(idx, Array):
            if not inserted_indirection_shape:
                lhs_shape += indirection_shape
                inserted_indirection_shape = True

            if isinstance(axis_len, int):
                name = vng("in")
                rhs_indices.append(prim.Subscript(
                                     prim.Variable(name),
                                     get_indexing_expression(idx.shape,
                                                             ((1,)*indirection_start
                                                              + indirection_shape)))
                                   % axis_len)
                bindings[name] = idx
            else:
                raise NotImplementedError
        else:
            raise AssertionError

    name = vng("in")
    bindings[name] = ary

    return IndexLambda(expr=prim.Subscript(prim.Variable(name),
                                           tuple(rhs_indices)),
                       shape=tuple(lhs_shape),
                       dtype=ary.dtype,
                       bindings=bindings)


# }}}
