__doc__ = """
.. currentmodule:: pytato.tags


Pre-Defined Tags
----------------

.. autoclass:: ImplementAs
.. autoclass:: ImplementationStrategy
.. autoclass:: CountNamed
.. autoclass:: ImplStored
.. autoclass:: ImplInlined
.. autoclass:: ImplDefault
"""


from pytools.tag import Tag, UniqueTag, tag_dataclass
from typing import Optional


# {{{ pre-defined tag: ImplementAs

@tag_dataclass
class ImplementationStrategy(Tag):
    """
    See :class:`ImplementAs`.
    """


@tag_dataclass
class ImplStored(ImplementationStrategy):
    """
    Tagged to a :class:`pytato.Array` expression that's to be materialized.

    .. attribute:: prefix

        The allocated variable would bear a name prefixed-by *prefix*
    """
    prefix: Optional[str] = None


@tag_dataclass
class ImplInlined(ImplementationStrategy):
    pass


@tag_dataclass
class ImplDefault(ImplementationStrategy):
    pass


@tag_dataclass
class ImplementAs(UniqueTag):
    """
    Records metadata to be attached to :class:`pytato.Array` to convey
    information to a :class:`pytato.target.Target` on how its supposed to be
    lowered to.

    .. attribute:: strategy

        An instance of :class:`ImplementationStrategy`.
    """

    strategy: ImplementationStrategy

# }}}


# {{{ pre-defined tag: CountNamed

@tag_dataclass
class CountNamed(UniqueTag):
    """
    .. attribute:: name
    """

    name: str

# }}}
