import tensorflow
from tensorflow.keras import backend as K
from utils.CustomOperations.ternary_ops import ternarize
def ternary_tanh(x):
    x = K.clip(x, -1, 1)
    return ternarize(x)