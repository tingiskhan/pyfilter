# TODO: When searchsorted is implemented - add this code
import torch


def set_default_tensor(dtype):
    """
    Sets the default tensor type
    :type dtype: str
    :return: None
    """

    torch.set_default_tensor_type(dtype)
