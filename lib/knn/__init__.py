import unittest
import gc
import operator as op
import functools
import torch
from torch.autograd import Variable, Function
from lib.knn import knn_pytorch as knn_pytorch

class KNearestNeighbor(Function):
  """ Compute k nearest neighbors for each query point.
  """
  # def __init__(self, k):
  #   self.k = k

  @staticmethod
  def forward(ctx, ref, query):
    device = query.device
    ref = ref.float().to(device)
    query = query.float().to(device)

    inds = torch.empty(query.shape[0], 1, query.shape[2]).long().to(device)

    if hasattr(knn_pytorch, 'knn'):
      knn_pytorch.knn(ref, query, inds)
    else:
      # Fallback for environments where the legacy CUDA extension is not built.
      # The historical extension returns 1-based indices.
      distances = torch.cdist(query.transpose(1, 2), ref.transpose(1, 2))
      inds.copy_(distances.argmin(dim=2).add(1).unsqueeze(1).long())

    return inds


class TestKNearestNeighbor(unittest.TestCase):

  def test_forward(self):
    knn = KNearestNeighbor(2)
    while(1):
        D, N, M = 128, 100, 1000
        ref = Variable(torch.rand(2, D, N))
        query = Variable(torch.rand(2, D, M))

        inds = knn(ref, query)
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                print(functools.reduce(op.mul, obj.size()) if len(obj.size()) > 0 else 0, type(obj), obj.size())
        #ref = ref.cpu()
        #query = query.cpu()
        print(inds)


if __name__ == '__main__':
  unittest.main()
