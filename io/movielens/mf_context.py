from __future__ import division
from __future__ import print_function

from matrix_factorization import MatrixFactorization

from contextual.context import Context


class MfContext(Context):
    def __init__(self, data_path=None):
            mf = MatrixFactorization(data_path)

            #TODO: Correct?
            self._context_data = {str(i): mf.Q[i] for i in range(len(mf.Q))}

    @property
    def context_data(self):
        return self._context_data, len(self._context_data[str(0)])
