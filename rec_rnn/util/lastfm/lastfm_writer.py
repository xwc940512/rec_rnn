from __future__ import division

import os
import unittest


def main():
    split_data(os.path.join('../../data/lastfm', 'userid-timestamp-artid-artname-traid-traname.tsv'))
    #split_data(os.path.join('data_sets/data/ml-100k', 'u.data'))


def split_data(path):
    num_lines = file_len(path)
    with open(path) as f:
        train = []
        val = []
        test = []
        i = 1
        for line in f:
            if i / num_lines < .6:
                train.append(line)
            elif i / num_lines < .8:
                val.append(line)
            else:
                test.append(line)

            i += 1

        _write_to_file(train, "../../data/lastfm/lastfm_train.dat")
        _write_to_file(val, "../../data/lastfm/lastfm_val.dat")
        _write_to_file(test, "../../data/lastfm/lastfm_test.dat")


def _write_to_file(data, path):
    f = open(path, "w")
    for item in data:
        f.write("%s" % item)
    f.close()


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


class LSTMTest(unittest.TestCase):
    def test_lstm(self):
        main()


if __name__ == "__main__":
    unittest.main()
