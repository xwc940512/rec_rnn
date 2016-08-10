from __future__ import division

import collections
import os
from random import randint

from rec_rnn.util import util
from rec_rnn.util.reader import Reader


class LastfmReader(Reader):
    def __init__(self):
        self._data_path = None

    # TODO: Differentiate between same song name by different artists
    def raw_data(self, data_path=None):
        train_path = os.path.join(data_path, 'lastfm_train.dat')
        val_path = os.path.join(data_path, 'lastfm_val.dat')
        test_path = os.path.join(data_path, 'lastfm_test.dat')

        songs, users = LastfmReader._read(train_path)
        songs, users = LastfmReader._random_subset(songs, users, 2)
        song_ids = LastfmReader._generate_ids(songs)
        user_ids = LastfmReader._generate_ids(users)

        train_ids = LastfmReader._file_to_ids(train_path, song_ids, user_ids)
        val_ids = LastfmReader._file_to_ids(val_path, song_ids, user_ids)
        test_ids = LastfmReader._file_to_ids(test_path, song_ids, user_ids)

        return train_ids, val_ids, test_ids, len(util.unique(train_ids[0] + val_ids[0] + test_ids[0])) + 1, len(
            util.unique(train_ids[1] + val_ids[1] + test_ids[1])) + 1

    @staticmethod
    def _generate_ids(target):
        counter = collections.Counter(target)

        sort = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        songs = [line[0] for line in sort]
        song_to_id = dict(zip(songs, range(len(songs))))

        return song_to_id

    @staticmethod
    def _read(path):
        with open(path) as f:
            song_names = []
            users = []
            for line in f:
                split = line.split("\t")
                users.append(split[0])
                song_names.append(split[5].replace("\n", ""))
            return song_names, users

    @staticmethod
    def _random_subset(songs, users, num_users):
        sub_users = []
        sub_songs = []

        for i in range(num_users):
            rand = randint(0, len(users))
            # First appearance of random user
            index = users.index(users[rand])

            if sub_users.count(users[index]) == 0:
                id = users[index]
                while True:
                    current_id = users[index]
                    if current_id == id:
                        sub_users.append(id)
                        sub_songs.append(songs[index])
                    else:
                        break

                    index += 1
            else:
                i -= 1

        return sub_songs, sub_users


    @staticmethod
    def _file_to_ids(filename, song_to_id, user_to_id):
        songs, users = LastfmReader._read(filename)

        result_i = []
        result_u = []
        for i in range(len(songs)):
            try:
                result_i.append(song_to_id[songs[i]])
                result_u.append(user_to_id[users[i]])
            except KeyError:
                if len(result_i) > len(result_u):
                    del result_i[-1]

        return result_i, result_u

    @property
    def data_path(self):
        return self._data_path

