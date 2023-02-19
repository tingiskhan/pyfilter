from math import ceil
from typing import Dict, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from statsmodels.nonparametric.kde import KDEUnivariate

from .context import InferenceContext
from .sequential.state import SequentialAlgorithmState


def _do_plot(v: np.ndarray, w: np.ndarray, ax_, name, handled, **kwargs):
    """
    Utility function for plotting.
    """

    kde = KDEUnivariate(v)
    kde.fit(weights=w, fft=False)

    x_linspace = np.linspace(v.min(), v.max(), 250)

    ax_.plot(x_linspace, kde.evaluate(x_linspace), **kwargs)

    ax_.spines["top"].set_visible(False)
    ax_.spines["right"].set_visible(False)
    ax_.spines["left"].set_visible(False)
    ax_.axes.get_yaxis().set_visible(False)
    ax_.set_title(name)

    handled.append(ax_)


# TODO: Pretty ugly...
def mimic_arviz_posterior(
    context: InferenceContext,
    state: SequentialAlgorithmState,
    num_cols: int = 3,
    ax: Axes = None,
    constrained: Union[bool, Dict[str, bool]] = True,
    **kwargs,
) -> Axes:
    """
    Helper function for mimicking arviz plotting functionality.

    Args:
        context (InferenceContext): parameter context to plot.
        state (SequentialAlgorithmState): associated algorithm state.
        num_cols (int): number of columns.
        ax (Axes, optional): pre-defined axes to use.
    """

    if not isinstance(constrained, dict):
        constrained = {k: constrained for k in context.parameters.keys()}

    for name in context.parameters.keys():
        if name not in constrained:
            constrained[name] = True

    if ax is None:
        num_rows = ceil(sum(p.prior.get_numel(constrained[n]) for n, p in context.parameters.items()) / num_cols)
        _, ax = plt.subplots(num_rows, num_cols)

    flat_axes = ax.ravel()
    weights = state.normalized_weights().cpu().numpy()

    handled = []
    fig_index = 0

    for name, parameter in context.parameters.items():
        numel = parameter.prior.get_numel(constrained[name])

        if not constrained[name]:
            parameter = parameter.prior.get_unconstrained(parameter)

        flattened = parameter.view(-1, numel).cpu().numpy()
        for i in range(numel):
            ax_ = flat_axes[fig_index]

            title = f"{name}_{i:d}" if numel > 1 else name
            _do_plot(flattened[..., i], weights, ax_, title, handled, **kwargs)
            fig_index += 1

    for ax_ in (ax_ for ax_ in flat_axes if ax_ not in handled):
        ax_.axis("off")

    return ax
