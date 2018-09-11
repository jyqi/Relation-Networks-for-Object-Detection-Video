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

    def forward(self, is_train, req, in_data, out_data, aux):

        embs      = in_data[0]
        labels    = in_data[1]
        print('debuggin')

        # check if input embs is batch > 1 and check labels
        # it likely isn't so will need to use memory to store
        # keep checking memory for fullness and when full give a loss value other than 0
        # that is do nothing until the memory is filled

        # need to also work out how to use the dets here, which ones we ignore etc.


        # per_roi_loss_cls = mx.nd.SoftmaxActivation(cls_score) + 1e-14
        # per_roi_loss_cls = per_roi_loss_cls.asnumpy()
        # per_roi_loss_cls = per_roi_loss_cls[np.arange(per_roi_loss_cls.shape[0], dtype='int'), labels.astype('int')]
        # per_roi_loss_cls = -1 * np.log(per_roi_loss_cls)
        # per_roi_loss_cls = np.reshape(per_roi_loss_cls, newshape=(-1,))
        #
        # per_roi_loss_bbox = bbox_weights * mx.nd.smooth_l1((bbox_pred - bbox_targets), scalar=1.0)
        # per_roi_loss_bbox = mx.nd.sum(per_roi_loss_bbox, axis=1).asnumpy()
        #
        # top_k_per_roi_loss = np.argsort(per_roi_loss_cls + per_roi_loss_bbox)
        # labels_ohem = labels
        # labels_ohem[top_k_per_roi_loss[::-1][self._roi_per_img:]] = -1
        # bbox_weights_ohem = bbox_weights.asnumpy()
        # bbox_weights_ohem[top_k_per_roi_loss[::-1][self._roi_per_img:]] = 0
        #
        # labels_ohem = mx.nd.array(labels_ohem)
        # bbox_weights_ohem = mx.nd.array(bbox_weights_ohem)

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
