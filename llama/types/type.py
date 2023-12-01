from llama.types.base_specification import BaseSpecification
from llama.program.value import Value

from pydantic import PrivateAttr
import inspect
import typing


class Type(BaseSpecification):
    _value = PrivateAttr()

    def __init__(self, *args, **kwargs):
        if any_values(args, kwargs):
            unvalidated = super().construct(*args, **kwargs)
            object.__setattr__(self, "__dict__", unvalidated.__dict__)
            object.__setattr__(self, "__fields_set__", unvalidated.__fields_set__)
        else:
            super().__init__(*args, **kwargs)

        self._value = Value(type(self), data=self)

    def __getattribute__(self, name):
        if name.find("_") == 0:
            if name != "_value":
                return super().__getattribute__(name)

        # fix access to private fields
        members = inspect.getmembers(Type)
        for member_name, member in members:
            if member_name == "__dict__":
                value = member["_value"].__get__(self)

        if name == "_value":
            return value

        if name in self.__dict__:
            return value._get_field(name)

        return super().__getattribute__(name)

    def _get_attribute_raw(self, name):
        return super().__getattribute__(name)

    @classmethod
    def _get_field_type(cls, name):
        field_to_type = typing.get_type_hints(cls)
        return field_to_type[name]

    def __getitem__(self, name):
        return getattr(self, name)


def any_values(args, kwargs):
    for arg in args:
        if isinstance(arg, Value):
            return True

    for arg in kwargs.values():
        if isinstance(arg, Value):
            return True

    return False
