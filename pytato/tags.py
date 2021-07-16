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


# {{{ pre-defined tag: ImplementAs

@tag_dataclass
class ImplementationStrategy(Tag):
    pass


@tag_dataclass
class ImplStored(ImplementationStrategy):
    pass


@tag_dataclass
class ImplInlined(ImplementationStrategy):
    pass


@tag_dataclass
class ImplDefault(ImplementationStrategy):
    pass


@tag_dataclass
class ImplementAs(UniqueTag):
    """
    .. attribute:: strategy
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
