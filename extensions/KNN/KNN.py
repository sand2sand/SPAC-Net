import os
import torch
import torch.nn as nn
import importlib
import KNN_CUDA as _knn


def knn(ref, query, k):
    d, i = _knn.knn(ref, query, k)
    i -= 1
    return d, i


def _T(t, mode=False):
    if mode:
        return t.transpose(0, 1).contiguous()
    else:
        return t


class KNN(nn.Module):

    def __init__(self, k, transpose_mode=False):
        super(KNN, self).__init__()
        self.k = k
        self._t = transpose_mode

    def forward(self, ref, query):
        assert ref.size(0) == query.size(0), "ref.shape={} != query.shape={}".format(ref.shape, query.shape)
        with torch.no_grad():
            batch_size = ref.size(0)
            D, I = [], []
            for bi in range(batch_size):
                r, q = _T(ref[bi], self._t), _T(query[bi], self._t)
                d, i = knn(r.float(), q.float(), self.k)
                d, i = _T(d, self._t), _T(i, self._t)
                D.append(d)
                I.append(i)
            D = torch.stack(D, dim=0)
            I = torch.stack(I, dim=0)
        return D, I


if __name__ == '__main__':
    import torch
    data1 = torch.randn(4, 3, 20).cuda()
    data2 = torch.randn(4, 3, 20).cuda()
    Knn = KNN(k=8, transpose_mode=False)
    _, idx = Knn(data1, data2)
    print(idx.shape)

