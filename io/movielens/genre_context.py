import os

from contextual.context import Context


class GenreContext(Context):
    def __init__(self, data_path=None):
        data_path = os.path.join(data_path, 'u.item')

        with open(data_path) as f:
            data = f.read().split("\n")
            genres = {element.split("|")[0]: [float(e) for e in element.split("|")[6:24]] for element in data}
            genres.pop('')

            self._context = genres
            self._context_state = self._context

    @property
    def context_data(self):
        return self._context, len(self._context[str(1)])
