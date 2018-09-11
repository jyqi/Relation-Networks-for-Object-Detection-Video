# --------------------------------------------------------
#
# --------------------------------------------------------

"""
Repmet Operator calcs probs for emb vectors
"""

import mxnet as mx
import numpy as np
from distutils.util import strtobool


class RepmetOperator(mx.operator.CustomOp):
    def __init__(self, n, m, k, emb_size):
        super(RepmetOperator, self).__init__()
        self._n = n
        self._m = m
        self._k = k
        self._emb_size = emb_size
        self._mem = mx.nd.zeros((self._m, self._n, self._emb_size))
        self._reps = mx.nd.zeros((self._m, self._k, self._emb_size))
        self.index = 0

    def forward(self, is_train, req, in_data, out_data, aux):

        embs      = in_data[0]
        labels    = in_data[1]
        self._mem[self.index] = embs[0][0]
        self.index += 1
        print(self._mem)

        print('debuggin')

        # assert batch size is 1, we will use mem to store


        # which detections/anchors do we ignore
        # option 1: ignore all that don't overlap with particular single instance, and have keep rest as other as usual
        #           feature is avg of these boxes

        # option 2: ignore all of different class, allowing multiple instances on same class per image
        #           feature is mean of these boxes

        # option 3: take the best overlapping box rather than mean them

        # ^^^^^^^^^^^^ These flags should be passed in? and decided in batch formations
        #              this can allow us to select individual instances and leave them out of future batches

        # what to do with 'other class'? what did we do before?

        # filter to get the box/boxes

        # avg the boxes

        # check memory for fullness
        # when full calc loss and return
        # otherwise let's just add to the mem, and return 0

        # is loss per box or single value?

        loss = in_data[1]

        for ind, val in enumerate([loss]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)


@mx.operator.register('Repmet')
class RepmetOperatorProp(mx.operator.CustomOpProp):
    def __init__(self, n, m, k, emb_size):
        super(RepmetOperatorProp, self).__init__(need_top_grad=False)  # false cause is a loss layer
        self._n = int(n)
        self._m = int(m)
        self._k = int(k)
        self._emb_size = int(emb_size)

    def list_arguments(self):
        return ['embs', 'labels']

    def list_outputs(self):
        return ['loss']

    def infer_shape(self, in_shape):
        return in_shape, [in_shape[1]]

    def create_operator(self, ctx, shapes, dtypes):
        return RepmetOperator(self._n, self._m, self._k, self._emb_size)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
