from typing import Dict, Any


_OBJTYPENAME = "objtype"


class Module(object):
    def state_dict(self) -> Dict[str, Any]:
        res = dict()
        res[_OBJTYPENAME] = self.__class__.__name__

        res.update(**self.populate_state_dict())

        return res

    def populate_state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def load_state_dict(self, state: Dict[str, Any]):
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
