r"""
Riemannian Manifold
"""

import abc


class RiemannianManifold(abc.ABC):
    """Riemannian Manifold"""

    @abc.abstractmethod
    def exp(self, p, v):
        r"""Exponential map
        $$exp_p(v)$$ map a point in tangent space to a point on the manifold
        $$
        T_p M \to M
        $$
        """
        raise NotImplementedError

    @abc.abstractmethod
    def log(self, p, q):
        r"""Logarithm map
        $$log_p(q)$$ map a point on the manifold to a point in tangent space
        $$
        M \to T_p M
        $$
        """
        raise NotImplementedError

    @abc.abstractmethod
    def random_tangent(self, p, random_type="gaussian", std=1.0):
        r"""Sample noise in the tangent space at point p
        $$T_p M$$
        """
        raise NotImplementedError

    @abc.abstractmethod
    def metric(self, p, v, w):
        r"""Inner product
        $$<v, w>_p$$ is the inner product of $$v$$ and $$w$$ at point $$p$$
        """
        raise NotImplementedError

    @abc.abstractmethod
    def grad(self, f, p):
        r"""Gradient
        $$\nabla_p f$$ is the gradient of $$f$$ at point $$p$$
        """
        raise NotImplementedError
