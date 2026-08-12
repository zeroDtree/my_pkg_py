r"""
Lie Group
"""

from .riemannian_manifold import RiemannianManifold


class LieGroup(RiemannianManifold):
    r"""Lie Group"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def multiply(self, p, q):
        r"""Multiply in Group"""
        raise NotImplementedError

    def inverse(self, p):
        r"""Inverse in Group"""
        raise NotImplementedError

    def identity(self):
        r"""Identity in Group"""
        raise NotImplementedError

    def left_translation(self, g, h):
        r"""
        $$L_g(h) = g \cdot h$$
        """
