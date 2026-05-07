import torch
import numpy as np
from sklearn.decomposition import PCA
import scipy.stats

from src.view.figure import Figure
from src.quantizers import iter_trainable_params, tensor_to_blocks
from src.quantizers.momos2d import tensor2D_to_blocks
from src.model import MLP


def extract_blocks(model, block_size):
    # differ from src.quantizers.momos._get_model_blocks because doesn't
    # concatenate the blocks
    block_size = int(block_size)
    with torch.no_grad():
        layer_specs = []
        all_blocks = []

        for param in iter_trainable_params(model):
            blocks, n_params, shape = tensor_to_blocks(param.detach(), block_size)
            layer_specs.append((param, int(blocks.size(0)), int(n_params), shape))
            all_blocks.append(blocks)

    return all_blocks, layer_specs


def load_model(checkpoint_path):
    # Use weights_only=False fallback because PL checkpoints contain non-tensor fields.
    try:
        weights = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    except Exception:
        weights = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    try:
        state_dict = weights["state_dict"]
    except Exception:
        state_dict = weights
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    model = MLP(3 * 32 * 32, 10)
    model.load_state_dict(new_state_dict)
    return model


def extract_blocks_2d(model, rows, cols):
    with torch.no_grad():
        layer_specs = []
        all_blocks = []
        for param in iter_trainable_params(model):
            blocks, n_params, shape = tensor2D_to_blocks(param.detach(), rows, cols)
            layer_specs.append((param, int(blocks.size(0)), int(n_params), shape))
            all_blocks.append(blocks)
    return all_blocks, layer_specs


def scatter_data(blocks, layers_specs, name):
    scatter_per_layer = {}
    scatter_motifs_per_layer = {}

    for i, (b, _) in enumerate(zip(blocks, layers_specs)):
        uniques_motifs_l = b.unique(dim=0)
        scatter_per_layer[f"Layer {i + 1}"] = uniques_motifs_l.T
        if i % 2 == 0 and i < 7:
            scatter_motifs_per_layer[f"Layer {i + 1} with {name}"] = {
                "scatter": uniques_motifs_l.T
            }

    return (
        scatter_per_layer,
        scatter_motifs_per_layer,
    )


def correlation_data(all_blocks: torch.Tensor) -> dict:
    """Compute Pearson and Spearman correlation between x_1 and x_2 columns."""
    x1 = all_blocks[:, 0].numpy()
    x2 = all_blocks[:, 1].numpy()
    pearson_r, pearson_p = scipy.stats.pearsonr(x1, x2)
    spearman_r, spearman_p = scipy.stats.spearmanr(x1, x2)
    return {
        "x1": x1,
        "x2": x2,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }


def report_weight_distribution(
    run,
    frequencies=None,
    norms=None,
    scatter=None,
    scatter_layer=None,
) -> list[Figure]:
    res = []
    if scatter:
        fig = Figure(fontsize=17)
        fig.plot(
            scatter,
            f"Motifs per layer - {run[2]}",
            symbol="o",
            axis=None,
            x_label="$X_1$",
            y_label="$X_2$",
        )
        res.append(fig)

    if frequencies:
        fig = Figure(fontsize=17)
        fig.plot(
            frequencies,
            f"Motifs frequencies - {run[2]}",
            x_label="Motifs",
            y_label="Frequency",
            colors=["red", "green", "pink", "coral", "purple"],
            logy=True,
            axis=None,
        )
        res.append(fig)

    if norms:
        fig = Figure(fontsize=17)
        fig.plot(
            norms,
            f"Motifs' norms per layer - {run[2]}",
            x_label="Norms",
            y_label="Frequency",
            logy=True,
            axis=None,
            symbol="x",
        )
        res.append(fig)

    if scatter_layer:
        fig = Figure(
            f"Motifs' norms per layer - {run[2]}",
            nrows=2,
            ncols=2,
        )
        for (n, d), c in zip(scatter_layer.items(), ["red", "green", "pink", "coral"]):
            _ = fig.plot(
                d,
                n,
                legend=False,
                symbol="o",
                axis=None,
                x_label="$X_1$",
                y_label="$X_2$",
                colors=[c],
            )
        res.append(fig)

    return res


def _plot_correlation(corr: dict, epoch: str) -> Figure:
    """Scatter of all (x1, x2) pairs with regression line and correlation stats."""
    x1 = corr["x1"]
    x2 = corr["x2"]
    pearson_r = corr["pearson_r"]
    pearson_p = corr["pearson_p"]
    spearman_r = corr["spearman_r"]
    spearman_p = corr["spearman_p"]

    fig = Figure(fontsize=17)
    scatter_data_dict = {"Weight Correlation": np.array([x1, x2])}
    fig.plot(
        scatter_data_dict,
        f"Weight Correlation (epoch {epoch})",
        symbol="o",
        axis=None,
        x_label="$X_1$",
        y_label="$X_2$",
    )

    # Add regression line and stats as text annotation
    ax = fig._ax()
    z = np.polyfit(x1, x2, 1)
    p = np.poly1d(z)
    x_line = np.array([x1.min(), x1.max()])
    ax.plot(x_line, p(x_line), "r--", linewidth=2, label="Linear fit")

    stats_text = f"Pearson r={pearson_r:.3f} (p={pearson_p:.2e})\nSpearman r={spearman_r:.3f} (p={spearman_p:.2e})"
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.legend()

    return fig


def _plot_all_blocks(run, all_blocks, blocks, layers_specs) -> list[Figure]:
    if all_blocks.shape[1] == 2:
        _, counts = all_blocks.unique(dim=0, return_counts=True)

        sort_idx = torch.argsort(counts, descending=True)
        frequencies = {
            "All layers": (range(len(sort_idx)), counts[sort_idx] / len(counts))
        }

        modules = module_data(all_blocks, blocks)
        scatter, scatter_layer = scatter_data(blocks, layers_specs, run[2])

        figures = report_weight_distribution(
            run, frequencies, modules, scatter, scatter_layer
        )

        corr = correlation_data(all_blocks)
        figures.append(_plot_correlation(corr, run[2]))

        return figures

    elif all_blocks.shape[1] > 2:
        pca = PCA(n_components=0.9999)
        pca = pca.fit(all_blocks)
        block_transformed = [pca.transform(b) for b in blocks]

        figures = []
        start = 0
        for end in range(2, all_blocks.shape[1] + 1, 2):
            blocks_cut = [b[:, start:end] for b in block_transformed]
            figures += plot_blocks(
                (run[0], run[1], str(run[2]) + f" Dim={end - 1, end}"),
                blocks_cut,
                layers_specs,
            )
            start = end

        frequencies = frequency_data(all_blocks, blocks)
        modules = module_data(all_blocks, blocks)
        figures += report_weight_distribution(
            run, frequencies=frequencies, norms=modules
        )

        return figures
    else:
        raise ValueError("What did you mean to do? In PlotWeight, shape is < 2")


def plot_weights(run) -> list[Figure]:
    model = load_model(run[0])
    blocks, layers_specs = extract_blocks(model, run[1])
    all_blocks = torch.cat(blocks, dim=0)
    return _plot_all_blocks(run, all_blocks, blocks, layers_specs)


def plot_weights_2d(run) -> list[Figure]:
    # run = (checkpoint_path, rows, cols, capacity, epoch)
    # rows/cols must be supplied by the caller (fetched from wandb run config).
    model = load_model(run[0])
    rows, cols = int(run[1]), int(run[2])
    epoch = run[4] if len(run) > 4 else "?"
    blocks, layers_specs = extract_blocks_2d(model, rows, cols)
    all_blocks = torch.cat(blocks, dim=0)
    run_display = (run[0], f"rows={rows},cols={cols}", f"epoch={epoch}")
    return _plot_all_blocks(run_display, all_blocks, blocks, layers_specs)


def plot_blocks(run, blocks, layers_specs):
    scatter, scatter_layer = scatter_data_numpy(blocks, layers_specs, run[2])
    return report_weight_distribution(run, scatter=scatter, scatter_layer=scatter_layer)


def scatter_data_numpy(blocks, layers_specs, capacity):
    scatter_per_layer = {}
    scatter_motifs_per_layer = {}

    for i, (b, _) in enumerate(zip(blocks, layers_specs)):
        uniques_motifs_l = np.unique(b, axis=1)
        scatter_per_layer[f"Layer {i + 1}"] = uniques_motifs_l.T
        if i % 2 == 0 and i < 7:
            scatter_motifs_per_layer[f"Layer {i + 1} with n_params={capacity}"] = {
                "scatter": uniques_motifs_l.T
            }

    return (
        scatter_per_layer,
        scatter_motifs_per_layer,
    )


def frequency_data(all_blocks, blocks):
    motifs, inverse_indices, counts = all_blocks.unique(
        dim=0, return_inverse=True, return_counts=True
    )

    sort_idx = torch.argsort(counts, descending=True)

    remapper = torch.zeros(len(motifs), dtype=torch.long)
    remapper[sort_idx] = torch.arange(len(motifs))
    sorted_inverse_indices = remapper[inverse_indices]

    # 4. Split the sorted inverse indices back into layers
    layer_sizes = [b.size(0) for b in blocks]
    per_layer_inverse = torch.split(sorted_inverse_indices, layer_sizes)

    # 5. Compute counts for each layer
    frequencies = {}
    num_motifs = len(motifs)

    n = 0
    for i, l_inv in enumerate(per_layer_inverse[:-3]):
        if i % 2 == 0:
            layer_histogram = torch.bincount(l_inv, minlength=num_motifs)

            n += layer_histogram.sum()

            frequencies[f"Layer {i + 1}"] = (
                range(len(layer_histogram)),
                layer_histogram / layer_histogram.sum() * 100,
            )

    print(counts.sum())
    print(n)
    frequencies["All layers"] = (
        range(len(counts)),
        counts[sort_idx] / counts.sum() * 100,
    )

    return frequencies


def module_data(all_blocks, blocks):
    all_blocks_norm = torch.linalg.vector_norm(all_blocks, ord=2, dim=1)
    blocks_norm = [torch.linalg.vector_norm(b, ord=2, dim=1) for b in blocks]

    modules, counts = all_blocks_norm.unique(return_counts=True)
    counts.sort()

    norms = {}
    norms["All layers"] = (
        modules,
        counts / counts.sum() * 100,
    )

    for i, b in enumerate(blocks_norm):
        modules, counts = b.unique(dim=0, return_counts=True)

        c, _ = torch.sort(counts)

        norms[f"Layer {i + 1}"] = (
            modules,
            c / c.sum() * 100,
        )

    return norms
