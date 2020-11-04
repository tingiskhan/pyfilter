from typing import Type, Dict


_OBJTYPENAME = 'objtype'


def _find_types(x, type_: Type) -> Dict[str, object]:
    """
    Helper method for finding all type_ in x.
    :param x: The object
    :return: Dictionary
    """

    return {k: v for k, v in vars(x).items() if isinstance(v, type_)}


class Module(object):
    def _find_obj_helper(self, type_: Type):
        """
        Helper object for finding a specific type of objects in self.
        :param type_: The type to filter on
        """

        return _find_types(self, type_)

    def state_dict(self) -> Dict[str, object]:
        """
        Returns the state dictionary.
        """
        res = dict()
        res[_OBJTYPENAME] = self.__class__.__name__

        res.update(**self.populate_state_dict())

        return res

    def populate_state_dict(self) -> Dict[str, object]:
        raise NotImplementedError()

    def load_state_dict(self, state: Dict[str, object]):
        """
        Loads the state dictionary.
        :param state: The state dictionary
        :return: Self
        """

        if state[_OBJTYPENAME] != self.__class__.__name__:
            raise ValueError(f"Trying to load '{self.__class__.__name__}' from '{state[_OBJTYPENAME]}'!")

        for k, v in state.items():
            if k == _OBJTYPENAME:
                continue

            if k not in vars(self):
                raise ValueError(f"Could not find attribute '{k}' on self!")

            attribute = getattr(self, k)

            if isinstance(attribute, Module):
                attribute.load_state_dict(v)
            else:
                setattr(self, k, v)

        return self