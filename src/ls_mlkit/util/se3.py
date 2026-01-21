# Adapted from OpenFold
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import Tensor


# According to DeepMind, this prevents rotation compositions from being
# computed on low-precision tensor cores. I'm personally skeptical that it
# makes a difference, but to get as close as possible to their outputs, I'm
# adding it.
def rot_matmul(a, b):
    e = ...
    row_1 = torch.stack(
        [
            a[e, 0, 0] * b[e, 0, 0] + a[e, 0, 1] * b[e, 1, 0] + a[e, 0, 2] * b[e, 2, 0],
            a[e, 0, 0] * b[e, 0, 1] + a[e, 0, 1] * b[e, 1, 1] + a[e, 0, 2] * b[e, 2, 1],
            a[e, 0, 0] * b[e, 0, 2] + a[e, 0, 1] * b[e, 1, 2] + a[e, 0, 2] * b[e, 2, 2],
        ],
        dim=-1,
    )
    row_2 = torch.stack(
        [
            a[e, 1, 0] * b[e, 0, 0] + a[e, 1, 1] * b[e, 1, 0] + a[e, 1, 2] * b[e, 2, 0],
            a[e, 1, 0] * b[e, 0, 1] + a[e, 1, 1] * b[e, 1, 1] + a[e, 1, 2] * b[e, 2, 1],
            a[e, 1, 0] * b[e, 0, 2] + a[e, 1, 1] * b[e, 1, 2] + a[e, 1, 2] * b[e, 2, 2],
        ],
        dim=-1,
    )
    row_3 = torch.stack(
        [
            a[e, 2, 0] * b[e, 0, 0] + a[e, 2, 1] * b[e, 1, 0] + a[e, 2, 2] * b[e, 2, 0],
            a[e, 2, 0] * b[e, 0, 1] + a[e, 2, 1] * b[e, 1, 1] + a[e, 2, 2] * b[e, 2, 1],
            a[e, 2, 0] * b[e, 0, 2] + a[e, 2, 1] * b[e, 1, 2] + a[e, 2, 2] * b[e, 2, 2],
        ],
        dim=-1,
    )

    return torch.stack([row_1, row_2, row_3], dim=-2)


def rot_vec_mul(r, t):
    x = t[..., 0]
    y = t[..., 1]
    z = t[..., 2]
    return torch.stack(
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )


class T:
    def __init__(self, rots, trans):
        self.rots = rots
        self.trans = trans

        if self.rots is None and self.trans is None:
            raise ValueError("Only one of rots and trans can be None")
        elif self.rots is None:
            self.rots = T.identity_rot(self.trans.shape[:-1], self.trans.dtype, self.trans.device)
        elif self.trans is None:
            self.trans = T.identity_trans(self.rots.shape[:-2], self.rots.dtype, self.rots.device)

        if self.rots.shape[-2:] != (3, 3) or self.trans.shape[-1] != 3 or self.rots.shape[:-2] != self.trans.shape[:-1]:
            raise ValueError("Incorrectly shaped input")

    def __getitem__(self, index):
        if type(index) != tuple:
            index = (index,)
        return T(self.rots[index + (slice(None), slice(None))], self.trans[index + (slice(None),)])

    def __eq__(self, obj):
        return torch.all(self.rots == obj.rots) and torch.all(self.trans == obj.trans)

    def __mul__(self, right):
        rots = self.rots * right[..., None, None]
        trans = self.trans * right[..., None]

        return T(rots, trans)

    def __rmul__(self, left):
        return self.__mul__(left)

    def to(self, device):
        if isinstance(device, T):
            self.trans = self.trans.to(device.get_trans())
            self.rots = self.rots.to(device.get_rots())
        else:
            self.trans = self.trans.to(device)
            self.rots = self.rots.to(device)
        return self

    @property
    def shape(self):
        s = self.rots.shape[:-2]
        return s if len(s) > 0 else torch.Size([1])

    def get_trans(self):
        return self.trans

    def get_rots(self):
        return self.rots

    def compose(self, t):
        rot_1, trn_1 = self.rots, self.trans
        rot_2, trn_2 = t.rots, t.trans

        rot = rot_matmul(rot_1, rot_2)
        trn = rot_vec_mul(rot_1, trn_2) + trn_1

        return T(rot, trn)

    def apply(self, pts):
        r, t = self.rots, self.trans
        rotated = rot_vec_mul(r, pts)
        return rotated + t

    def invert_apply(self, pts):
        r, t = self.rots, self.trans
        pts = pts - t
        return rot_vec_mul(r.transpose(-1, -2), pts)

    def invert(self):
        rot_inv = self.rots.transpose(-1, -2)
        trn_inv = rot_vec_mul(rot_inv, self.trans)

        return T(rot_inv, -1 * trn_inv)

    def unsqueeze(self, dim):
        if dim >= len(self.shape):
            raise ValueError("Invalid dimension")
        rots = self.rots.unsqueeze(dim if dim >= 0 else dim - 2)
        trans = self.trans.unsqueeze(dim if dim >= 0 else dim - 1)

        return T(rots, trans)

    @staticmethod
    def identity_rot(shape, dtype, device, requires_grad=False):
        rots = torch.eye(3, dtype=dtype, device=device, requires_grad=requires_grad)
        rots = rots.view(*((1,) * len(shape)), 3, 3)
        rots = rots.expand(*shape, -1, -1)

        return rots

    @staticmethod
    def identity_trans(shape, dtype, device, requires_grad=False):
        trans = torch.zeros((*shape, 3), dtype=dtype, device=device, requires_grad=requires_grad)
        return trans

    @staticmethod
    def identity(shape, dtype, device, requires_grad=False):
        return T(
            T.identity_rot(shape, dtype, device, requires_grad),
            T.identity_trans(shape, dtype, device, requires_grad),
        )

    @staticmethod
    def from_4x4(t):
        rots = t[..., :3, :3]
        trans = t[..., :3, 3]
        return T(rots, trans)

    def to_4x4(self):
        tensor = torch.zeros((*self.shape, 4, 4), device=self.rots.device)
        tensor[..., :3, :3] = self.rots
        tensor[..., :3, 3] = self.trans
        tensor[..., 3, 3] = 1
        return tensor

    @staticmethod
    def from_tensor(t):
        return T.from_4x4(t)

    @staticmethod
    def rigid_from_3_points(x_1: Tensor, x_2: Tensor, x_3: Tensor, eps: float = 1e-8):
        v1 = x_3 - x_2
        v2 = x_1 - x_2
        e1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + eps)
        u2 = v2 - (torch.einsum("...li, ...li -> ...l", e1, v2)[..., None] * e1)
        e2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + eps)
        e3 = torch.cross(e1, e2, dim=-1)
        R = torch.cat([e1[..., None], e2[..., None], e3[..., None]], axis=-1)  # [B,L,3,3] - rotation matrix

        return T(R, x_2)

    @staticmethod
    def concat(ts, dim):
        rots = torch.cat([t.rots for t in ts], dim=dim if dim >= 0 else dim - 2)
        trans = torch.cat([t.trans for t in ts], dim=dim if dim >= 0 else dim - 1)

        return T(rots, trans)

    def map_tensor_fn(self, fn):
        """Apply a function that takes a tensor as its only argument to the
        rotations and translations, treating the final two/one
        dimension(s), respectively, as batch dimensions.

        E.g.: Given t, an instance of T of shape [N, M], this function can
        be used to sum out the second dimension thereof as follows:

            t = t.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

        The resulting object has rotations of shape [N, 3, 3] and
        translations of shape [N, 3]
        """
        rots = self.rots.view(*self.rots.shape[:-2], 9)
        rots = torch.stack(list(map(fn, torch.unbind(rots, -1))), dim=-1)
        rots = rots.view(*rots.shape[:-1], 3, 3)

        trans = torch.stack(list(map(fn, torch.unbind(self.trans, -1))), dim=-1)

        return T(rots, trans)

    def stop_rot_gradient(self):
        return T(self.rots.detach(), self.trans)

    def scale_translation(self, factor):
        return T(self.rots, self.trans * factor)


_quat_elements = ["a", "b", "c", "d"]
_qtr_keys = [l1 + l2 for l1 in _quat_elements for l2 in _quat_elements]
_qtr_ind_dict = {key: ind for ind, key in enumerate(_qtr_keys)}


def _to_mat(pairs):
    mat = torch.zeros((4, 4))
    for pair in pairs:
        key, value = pair
        ind = _qtr_ind_dict[key]
        mat[ind // 4][ind % 4] = value

    return mat


_qtr_mat = torch.zeros((4, 4, 3, 3))
_qtr_mat[..., 0, 0] = _to_mat([("aa", 1), ("bb", 1), ("cc", -1), ("dd", -1)])
_qtr_mat[..., 0, 1] = _to_mat([("bc", 2), ("ad", -2)])
_qtr_mat[..., 0, 2] = _to_mat([("bd", 2), ("ac", 2)])
_qtr_mat[..., 1, 0] = _to_mat([("bc", 2), ("ad", 2)])
_qtr_mat[..., 1, 1] = _to_mat([("aa", 1), ("bb", -1), ("cc", 1), ("dd", -1)])
_qtr_mat[..., 1, 2] = _to_mat([("cd", 2), ("ab", -2)])
_qtr_mat[..., 2, 0] = _to_mat([("bd", 2), ("ac", -2)])
_qtr_mat[..., 2, 1] = _to_mat([("cd", 2), ("ab", 2)])
_qtr_mat[..., 2, 2] = _to_mat([("aa", 1), ("bb", -1), ("cc", -1), ("dd", 1)])


def quat_to_rot(quat):  # [*, 4]
    # [*, 4, 4]
    quat = quat[..., None] * quat[..., None, :]

    # [*, 4, 4, 3, 3]
    shaped_qtr_mat = _qtr_mat.view((1,) * len(quat.shape[:-2]) + (4, 4, 3, 3))
    quat = quat[..., None, None] * shaped_qtr_mat.to(quat.device)

    # [*, 3, 3]
    return torch.sum(quat, dim=(-3, -4))
