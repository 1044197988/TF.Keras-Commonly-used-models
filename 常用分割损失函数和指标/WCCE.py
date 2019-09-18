from tensorflow.python import keras
from itertools import product
import numpy as np
from tensorflow.python.keras.utils import losses_utils
#weights->数组
class WeightedCategoricalCrossentropy(keras.losses.CategoricalCrossentropy):

    def __init__(
        self,
        weights,
        from_logits=False,
        label_smoothing=0,
        reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
        name='categorical_crossentropy',
    ):
        super().__init__(
            from_logits, label_smoothing, reduction, name=f"weighted_{name}"
        )
        self.weights = weights

    def call(self, y_true, y_pred):
        weights = self.weights
        nb_cl = len(weights)
        final_mask = keras.backend.zeros_like(y_pred[:, 0])
        y_pred_max = keras.backend.max(y_pred, axis=1)
        y_pred_max = keras.backend.reshape(
            y_pred_max, (keras.backend.shape(y_pred)[0], 1))
        y_pred_max_mat = keras.backend.cast(
            keras.backend.equal(y_pred, y_pred_max), keras.backend.floatx())
        for c_p, c_t in product(range(nb_cl), range(nb_cl)):
            final_mask += (
                weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
        return super().call(y_true, y_pred) * final_mask


from tensorflow.keras import backend as K
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss
