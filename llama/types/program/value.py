class Value(object):
    def __init__(self, type, data=None):
        self._type = type
        self._data = data
        self._function = None
        self._index = None

    def _get_field(self, name):
        if self._data is None:
            raise Exception("Value Access Error: must compute value before acessing")

        return self._data._get_attribute_raw(name)

    def __str__(self):
        if self._data is None:
            raise Exception("Value Access Error: must compute value before acessing")

        return str(self._data)

    def __int__(self):
        if self._data is None:
            raise Exception("Value Access Error: must compute value before acessing")

        return int(self._data)

    def __float__(self):
        if self._data is None:
            raise Exception("Value Access Error: must compute value before acessing")

        return float(self._data)

    def __gt__(self, other):
        if self._data is None:
            raise Exception("Value Access Error: must compute value before acessing")

        if isinstance(other, Value):
            other = other._get_data()

        return self._data > other

    def _get_data(self):
        if self._data is None:
            raise Exception("Value Access Error: must compute value before acessing")

        return self._data

    def __repr__(self):
        return str(self)

    def __getattribute__(self, name):
        return super().__getattribute__(name)

    def _get_attribute_raw(self, name):
        return super().__getattribute__(name)
