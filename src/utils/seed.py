import random

import numpy as np
from tensorflow.random import set_seed


def manual_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)
    set_seed(seed)
