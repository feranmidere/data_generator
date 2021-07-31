from . import dist_gen
from . import nn_gen

from .nn_gen import SequentialRegressionSynthesiser
from .dist_gen import SimpleClassificationSynthesiser

all = {
    'SRS': SequentialRegressionSynthesiser,
    'SCS': SimpleClassificationSynthesiser
}