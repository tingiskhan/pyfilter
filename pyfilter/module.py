import torch
from types import GeneratorType


_OBJTYPENAME = 'objtype'


class TensorContainer(object):
    def __init__(self, *args):
        self._cont = tuple(args) if not isinstance(args[0], GeneratorType) else tuple(*args)

    @property
    def tensors(self):
        """
        Returns the tensors
        :rtype: tuple[torch.Tensor]
        """

        return self._cont

    def append(self, x):
        if not isinstance(x, torch.Tensor):
            raise NotImplementedError()

        self._cont += (x,)

        return self

    def __getitem__(self, item):
        return self._cont[item]

    def __iter__(self):
        return (t for t in self._cont)


class Module(object):
    def _find_obj_helper(self, type_):
        """
        Helper object for finding a specific type of objects in self.
        :param type_: The type to filter on
        :type type_: object
        :rtype: dict[str, object]
        """

        return {k: v for k, v in vars(self).items() if isinstance(v, type_)}

    def modules(self):
        """
        Finds and returns all instances of type module.
        :rtype: tuple[Module]
        """

        return tuple(self._find_obj_helper(Module).values())

    def tensors(self):
        """
        Finds and returns all instances of type module.
        :rtype: tuple[torch.Tensor]
        """

        res = tuple()

        # ===== Find all tensor types ====== #
        res += tuple(self._find_obj_helper(torch.Tensor).values())

        # ===== Tensor containers ===== #
        for tc in self._find_obj_helper(TensorContainer).values():
            res += tc.tensors

        # ===== Modules ===== #
        for mod in self.modules():
            res += mod.tensors()

        return res

    def apply(self, f):
        """
        Applies function f to all tensors.
        :param f: The callable
        :type f: callable
        :return: Self
        :rtype: Module
        """

        for t in (t_ for t_ in self.tensors() if t_._base is None):
            t.data = f(t.data)

            if t._grad is not None:
                t._grad.data = f(t._grad.data)

        for t in (t_ for t_ in self.tensors() if t_._base is not None):
            t.data = t._base.data

        return self

    def to_(self, device):
        """
        Move to device.
        :param device: The device to move to
        :type device: str
        :return: Self
        :rtype: Module
        """

        return self.apply(lambda u: u.to(device))

    def state_dict(self):
        """
        Returns the state dictionary.
        :rtype: dict[str, object]
        """
        res = dict()
        res[_OBJTYPENAME] = self.__class__.__name__

        # ===== Tensors ===== #
        tens = self._find_obj_helper(torch.Tensor)
        res.update(tens)

        # ===== Tensor containers ===== #
        conts = self._find_obj_helper(TensorContainer)
        res.update(conts)

        # ===== Modules ===== #
        modules = self._find_obj_helper(Module)

        for k, m in modules.items():
            res[k] = m.state_dict()

        return res

    def load_state_dict(self, state):
        """
        Loads the state dictionary.
        :param state: The state dictionary
        :type state: dict
        :return: Self
        :rtype: Module
        """

        if state[_OBJTYPENAME] != self.__class__.__name__:
            raise ValueError(f'Cannot cast {state[_OBJTYPENAME]} as {self.__class__.__name__}!')

        for k, v in ((k_, v_) for k_, v_ in state.items() if k_ != _OBJTYPENAME):
            attr = getattr(self, k)

            if isinstance(attr, Module):
                attr.load_state_dict(v)

            setattr(self, k, v)

        return self
