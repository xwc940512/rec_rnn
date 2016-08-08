import os
import util

from operator import itemgetter
from contextual.reader import Reader


class MlReader(Reader):

    def __init__(self):
        self._data_path = None

    def raw_item_data(self, data_path=None):
        self._data_path = data_path
        train_path = os.path.join(data_path, 'ml_train.dat')
        val_path = os.path.join(data_path, 'ml_val.dat')
        test_path = os.path.join(data_path, 'ml_test.dat')

        train_movie_ids = MlReader._read_movie_ids(train_path)
        val_movie_ids = MlReader._read_movie_ids(val_path)
        test_movie_ids = MlReader._read_movie_ids(test_path)

        #TODO: Check & redo
        return train_movie_ids, val_movie_ids, test_movie_ids, len(util.unique(train_movie_ids + val_movie_ids + test_movie_ids)) + 1

    def raw_data(self, data_path=None):
        self._data_path = data_path
        train_path = os.path.join(data_path, 'ml_train.dat')
        val_path = os.path.join(data_path, 'ml_val.dat')
        test_path = os.path.join(data_path, 'ml_test.dat')

        train_ids = MlReader._read_all_ids(train_path)
        val_ids = MlReader._read_all_ids(val_path)
        test_ids = MlReader._read_all_ids(test_path)

        #TODO: Check & redo
        return train_ids, val_ids, test_ids, len(util.unique(train_ids[0] + val_ids[0] + test_ids[0])) + 1, len(util.unique(train_ids[1] + val_ids[1] + test_ids[1])) + 1


    @staticmethod
    def _read_all_ids(path):
        with open(path) as f:
            sorted_data = MlReader._sorted_data(f)

            return [element[1] for element in sorted_data], [element[0] for element in sorted_data]

    @staticmethod
    def _read_movie_ids(path):
        with open(path) as f:
            sorted_data = MlReader._sorted_data(f)

            return [element[1] for element in sorted_data]

    @staticmethod
    def _sorted_data(f):
        raw_data = f.read().split("\n")
        data = [element.split("\t") for element in raw_data]
        data.remove([''])
        return sorted(data, key=itemgetter(0, 3))

    @property
    def data_path(self):
        return self._data_path
