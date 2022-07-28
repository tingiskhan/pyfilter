import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from statsmodels.nonparametric.kde import KDEUnivariate
from math import ceil

from .context import ParameterContext
from .sequential.state import SequentialAlgorithmState


def _do_plot(v: torch.Tensor, w: torch.Tensor, ax_, name, handled, **kwargs):
    """
    Utility function for plotting.
    """

    v_numpy = v.cpu().numpy()

    kde = KDEUnivariate(v_numpy)
    kde.fit(weights=w, fft=False)

    x_linspace = np.linspace(v_numpy.min(), v_numpy.max(), 250)

    ax_.plot(x_linspace, kde.evaluate(x_linspace), **kwargs)

    ax_.spines["top"].set_visible(False)
    ax_.spines["right"].set_visible(False)
    ax_.spines["left"].set_visible(False)
    ax_.axes.get_yaxis().set_visible(False)
    ax_.set_title(name)

    handled.append(ax_)


def mimic_arviz_posterior(
        context: ParameterContext, state: SequentialAlgorithmState, num_cols: int = 3, ax: Axes = None, **kwargs
) -> Axes:
    """
    Helper function for mimicking arviz plotting functionality.

    Args:
        context: parameter context to plot.
        state: associated state.
        num_cols: the number of columns.
        ax: pre-defined axes to use.
    """

    if ax is None:
        num_rows = ceil(sum(p.prior.shape.numel() for p in context.parameters.values()) / num_cols)
        _, ax = plt.subplots(num_rows, num_cols)

    w = state.normalized_weights().cpu().numpy()
    flat_axes = ax.ravel()

    handled = list()
    fig_index = 0

    for p, v in context.parameters.items():
        if v.prior.get_numel() == 1:
            ax_ = flat_axes[fig_index]
            _do_plot(v, w, ax_, p, handled, **kwargs)
            fig_index += 1
        else:
            for i in range(v.shape[-1]):
                ax_ = flat_axes[fig_index]
                _do_plot(v[..., i], w, ax_, f"{p}_{i:d}", handled, **kwargs)
                fig_index += 1

    for ax_ in (ax_ for ax_ in flat_axes if ax_ not in handled):
        ax_.axis("off")

    return ax
