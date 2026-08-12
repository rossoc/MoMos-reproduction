"""Generic Pareto-front scatter, decoupled from any specific data source.

Consumes a plain DataFrame - it has no knowledge of Optuna, YAML, or
quantization - so callers translate their own data into columns first
(see `notebook/pareto_front_analysis.py` for the Optuna-specific reader).
"""

from typing import Literal, Sequence

import pandas as pd

from .figure import Figure

# Matplotlib marker codes, assigned to groups in sorted order. Extend this
# list if a study ever uses more distinct groups than it has entries.
MARKERS: list[str] = ["*", "o", "s", "^", "D", "v", "P", "X", "h", "<", ">", "8"]


def plot_pareto_front(
    data: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    optimal: str | None = None,
    x_label: str = "x",
    y_label: str = "y",
    title: str | None = None,
    markers: Sequence[str] = MARKERS,
    optimal_color: str = "crimson",
    dominated_color: str = "steelblue",
    dominated_alpha: float = 0.45,
    optimal_alpha: float = 0.95,
    markersize: float = 70.0,
    style: Literal["sci", "scientific", "plain"] = "plain",
    figsize: tuple[float, float] = (9, 7),
    fontsize: int = 13,
) -> Figure:
    """Scatter `data[x]` vs `data[y]`, one marker shape per distinct value of
    `data[group]` (cycled from `markers`). If `optimal` names a boolean
    column, Pareto-optimal rows are drawn solid/large and the rest
    faded/small, with a second color legend; otherwise every point is drawn
    the same way and only the method-shape legend is shown.

    Args:
        data: rows to plot - any DataFrame with at least `x`, `y`, `group`
            columns (and `optimal` if given). Not tied to any particular
            source; a hand-built DataFrame works just as well as one read
            from an Optuna study.
        x, y: column names for the scatter coordinates.
        group: column name whose distinct values each get their own marker
            shape (e.g. a quantization method).
        optimal: optional boolean column name marking Pareto-optimal rows.
        x_label, y_label, title: passed through to the underlying `Figure`.
        markers: marker codes assigned to `sorted(data[group].unique())` in
            order; must have at least as many entries as distinct groups.

    Returns:
        A `Figure` ready to `.save()` standalone or hand to a `Report` via
        `report.append_figures([fig])`.

    Raises:
        ValueError: more distinct `group` values than entries in `markers`.
    """
    groups = sorted(data[group].unique())
    if len(groups) > len(markers):
        raise ValueError(
            f"{len(groups)} distinct '{group}' values but only {len(markers)} "
            "markers in MARKERS - extend src/view/pareto.py:MARKERS."
        )
    marker_of = dict(zip(groups, markers))

    fig = Figure(fontsize=fontsize, figSize=figsize)
    ax = fig._ax()

    # (is_optimal, alpha, size, edgecolor) passes to draw per group. Without an
    # `optimal` column every row is drawn once, using the "optimal" styling.
    passes = (
        [(True, optimal_alpha, markersize, "black")]
        if optimal is None
        else [
            (True, optimal_alpha, markersize, "black"),
            (False, dominated_alpha, markersize * 0.6, "none"),
        ]
    )
    for g in groups:
        g_rows = data[data[group] == g]
        for is_opt, alpha, size, edge in passes:
            subset = g_rows if optimal is None else g_rows[g_rows[optimal] == is_opt]
            if subset.empty:
                continue
            ax.scatter(
                subset[x],
                subset[y],
                marker=marker_of[g],
                s=size,
                alpha=alpha,
                color=(optimal_color if is_opt else dominated_color),
                edgecolors=edge,
                linewidths=0.8,
            )

    # Legend #1: marker shape -> group. Proxy handles (empty scatters) so the
    # legend key shows the shape regardless of how many points were actually
    # drawn for that group.
    method_handles = [
        ax.scatter([], [], marker=marker_of[g], color="black", label=str(g))
        for g in groups
    ]
    method_legend = ax.legend(
        handles=method_handles, title="Method", loc="lower right", framealpha=1.0
    )

    # Legend #2 (only when `optimal` is given): color -> Pareto status. Added
    # via `add_artist` since a second `ax.legend()` call would otherwise
    # replace the first one.
    if optimal is not None:
        ax.add_artist(method_legend)
        status_handles = [
            ax.scatter([], [], marker="o", color=optimal_color, label="Pareto-optimal"),
            ax.scatter(
                [], [], marker="o", color=dominated_color, alpha=dominated_alpha,
                label="Dominated",
            ),
        ]
        ax.legend(handles=status_handles, loc="upper right", framealpha=1.0)

    fig._default_settings(x_label, y_label, title or "", style, False, False, grid=True)
    return fig
