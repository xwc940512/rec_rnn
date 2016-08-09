import numpy as np


class Reader(object):
    def raw_item_data(self, data_path=None):
        raise NotImplementedError("Abstract method")

    def raw_user_data(self, data_path=None):
        raise NotImplementedError("Abstract method")

    def item_iterator(self, input_data, batch_size, num_steps):
        input_data = np.array(input_data, dtype=np.int32)

        data_len = len(input_data)
        batch_len = data_len // batch_size
        data = np.zeros([batch_size, batch_len], dtype=np.int32)
        for i in range(batch_size):
            data[i] = input_data[batch_len * i:batch_len * (i + 1)]

        epoch_size = (batch_len - 1) // num_steps

        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

        for i in range(epoch_size):
            x = data[:, i*num_steps:(i+1)*num_steps]
            y = data[:, i*num_steps+1:(i+1)*num_steps+1]
            yield (x, y)

    def data_iterator(self, input_data, batch_size, num_steps):
        input_i = np.array(input_data[0], dtype=np.int32)
        input_u = np.array(input_data[1], dtype=np.int32)

        data_len = len(input_i)
        batch_len = data_len // batch_size
        data_i = np.zeros([batch_size, batch_len], dtype=np.int32)
        data_u = np.zeros([batch_size, batch_len], dtype=np.int32)
        for i in range(batch_size):
            data_i[i] = input_i[batch_len * i:batch_len * (i + 1)]
            data_u[i] = input_u[batch_len * i:batch_len * (i + 1)]

        epoch_size = (batch_len - 1) // num_steps

        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

        for i in range(epoch_size):
            x_i = data_i[:, i * num_steps:(i + 1) * num_steps]
            x_u = data_u[:, i * num_steps:(i + 1) * num_steps]
            y = data_i[:, i * num_steps + 1:(i + 1) * num_steps + 1]
            yield (x_i, x_u, y)




