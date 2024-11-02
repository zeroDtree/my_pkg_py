from __future__ import absolute_import


from .wrn import *
from .mlp import *


def get_network_mnist(network, **kwargs):

    networks = {"wrn": wrn, "mlp": mlp}
    return networks[network](**kwargs)
