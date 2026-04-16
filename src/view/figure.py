import matplotlib.pyplot as plt
from typing import Literal, Any
from matplotlib import font_manager
import pandas as pd

_FONT = font_manager.FontProperties(fname="note/fonts/HKGrotesk-Regular.ttf")

_COLORS = [
    "red",
    "blue",
    "green",
    "orange",
    "pink",
    "teal",
    "coral",
    "lavender",
    "purple",
    "gold",
    "lime",
    "navy",
    "crimson",
    "turquoise",
    "maroon",
    "olive",
    "indigo",
    "salmon",
    "sienna",
    "orchid",
    "khaki",
    "steelblue",
    "darkseagreen",
]


class Figure:
    def __init__(self, title=None, figSize=(12, 8), ncols=1, nrows=1, fontsize=13):
        self.figSize = figSize
        self.fontsize = fontsize
        self.ncols = ncols
        self.nrows = nrows

        # Adjust subplot parameters to prevent title/label overlap in multi-row figures
        if nrows > 1:
            plt.rcParams["figure.subplot.hspace"] = 0.35
            plt.rcParams["figure.subplot.bottom"] = 0.12

        self.fig, self.ax = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figSize, squeeze=False
        )
        self.plot_index = -1

        if (self.ncols > 1 or self.nrows > 1) and title is not None:
            assert isinstance(title, str)
            self.fig.suptitle(title, fontproperties=_FONT, fontsize=self.fontsize * 1.5)

    def _ax(self):
        row, col = divmod(self.plot_index, self.ncols)
        return self.ax[row][col]

    def _next_plot(self):
        self.plot_index += 1
        if self.plot_index >= self.ncols * self.nrows:
            raise RuntimeError(
                f"Too many plots on the current figure. Available axes: {self.ncols * self.nrows}, Current plot index: {self.plot_index}"
            )

    def _default_settings(self, x_label, y_label, exp_name, style, logx, logy):
        self._ax().legend()
        self._ax().set_xlabel(x_label, fontproperties=_FONT, fontsize=self.fontsize)
        self._ax().set_ylabel(y_label, fontproperties=_FONT, fontsize=self.fontsize)
        self._ax().ticklabel_format(
            axis="both", useMathText=True, useOffset=True, style=style, scilimits=(0, 0)
        )
        if self.ncols == 1 and self.nrows == 1:
            self.fig.suptitle(
                exp_name, fontproperties=_FONT, fontsize=self.fontsize * 1.5
            )
        else:
            self._ax().set_title(exp_name)

        self._ax().grid(True)
        self._ax().set_xlim(left=0)
        self._ax().set_ylim(bottom=0)

        if logx:
            self._ax().set_xscale("log")

        if logy:
            self._ax().set_yscale("log")

    def plot(
        self,
        data: dict[str, tuple[list[float], list[float]] | pd.DataFrame],
        exp_name: str,
        logx=False,
        logy=False,
        x_label="Epochs",
        y_label="Loss",
        symbol="-",
        style: Literal["sci", "scientific", "plain"] = "plain",
        figSize=(12, 8),
        markersize=5,
        skip_n=0,
        pop_n=0,
    ):
        self._next_plot()
        for i, (name, points) in enumerate(data.items()):
            if isinstance(points, pd.DataFrame):
                points = points.dropna().to_numpy().T
            self._ax().plot(
                points[0][skip_n : len(points[0]) - pop_n],
                points[1][skip_n : len(points[0]) - pop_n],
                symbol,
                markersize=markersize,
                label=name,
                color=_COLORS[i % len(_COLORS)],
            )

        self._default_settings(x_label, y_label, exp_name, style, logx, logy)

    def plot_with_var(
        self,
        data: dict[str, tuple[list[float], Any]],
        exp_name: str,
        logx=False,
        logy=False,
        x_label="Epochs",
        y_label="Loss",
        symbol="-",
        alpha=0.12,
        style: Literal["sci", "scientific", "plain"] = "plain",
        markersize=5,
    ):
        """
        data is of the kind:
        "run_name": (X, [[means...], [stds...]]), such that X are the epochs for
        example and the second array is a numpy array of shape (2, |X|)

        """
        self._next_plot()
        for i, (name, points) in enumerate(data.items()):
            self._ax().plot(
                points[0],
                points[1][0],
                symbol,
                markersize=markersize,
                label=name,
                color=_COLORS[i % len(_COLORS)],
            )
            self._ax().fill_between(
                points[0],
                points[1][0] - points[1][1],
                points[1][0] + points[1][1],
                alpha=alpha,
            )

        self._default_settings(x_label, y_label, exp_name, style, logx, logy)

    def plot3D(
        self,
        data: list[tuple[str, list[float]]],
        exp_name: str,
        cmap="viridis",
        levels=20,
        logx=1,
        logy=1,
        style: Literal["sci", "scientific", "plain"] = "plain",
    ):
        """
        Create a 3D plot from list of tuples (x, y, z), such that the z is
        indicated with a color.

        Args:
            data: List of tuples (x, y, z) where:
                x: x-axis value
                y: y-axis value
                z: z-axis value (height/accuracy)
            exp_name: Title for the plot
            cmap: Colormap for the contour plot (default: "viridis")
            levels: Number of contour levels (default: 20)
            logx: If >0, use log scale for x-axis with logx as base
            logy: If >0, use log scale for y-axis with logy as base
        """
        self._next_plot()
        ax = self._ax()

        assert len(data) == 3

        cntr = ax.tricontourf(
            data[0][1], data[1][1], data[2][1], cmap=cmap, levels=levels
        )

        cbar = plt.colorbar(cntr, ax=ax)
        cbar.set_label(data[2][0], fontproperties=_FONT, size=self.fontsize)

        if self.ncols == 1 and self.nrows == 1:
            self.fig.suptitle(
                exp_name, fontproperties=_FONT, fontsize=self.fontsize * 1.5
            )

        else:
            ax.set_title(exp_name, fontproperties=_FONT, fontsize=self.fontsize)

        ax.set_xlabel(data[0][0], fontproperties=_FONT, fontsize=self.fontsize)
        ax.set_ylabel(data[1][0], fontproperties=_FONT, fontsize=self.fontsize)
        ax.ticklabel_format(
            axis="both", useMathText=True, useOffset=True, style=style, scilimits=(0, 0)
        )
        ax.grid(True)

        if logx > 1:
            ax.set_xscale("log", base=logx)

        if logy > 1:
            ax.set_yscale("log", base=logy)

    def save(self, figure_name: str):
        import numpy as np

        if not hasattr(self, "ax_flat"):
            if isinstance(self.ax, np.ndarray):
                self.ax_flat = self.ax.flatten().tolist()
            elif isinstance(self.ax, list):
                self.ax_flat = self.ax
            else:
                self.ax_flat = [self.ax]

        if hasattr(self, "fig") and self.fig is not None:
            self.fig.savefig(f"{figure_name}.pdf", bbox_inches="tight")
        else:
            # Fallback: get the figure from the first axis in the list
            self.ax_flat[0].get_figure().savefig(
                f"{figure_name}.pdf", bbox_inches="tight"
            )

    def show(self):
        self.fig.show()
