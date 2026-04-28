import torch
import numpy as np
from sklearn.decomposition import PCA

from src.view.figure import Figure
from src.quantizers import iter_trainable_params, tensor_to_blocks
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
    weights = torch.load(
        checkpoint_path, weights_only=True, map_location=torch.device("cpu")
    )

    try:
        state_dict = weights["state_dict"]
    except Exception as _:
        state_dict = weights
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    model = MLP(3 * 32 * 32, 10)
    model.load_state_dict(new_state_dict)
    return model


def scatter_data(blocks, layers_specs, name):
    scatter_per_layer = {}
    scatter_motifs_per_layer = {}

    for i, (b, s) in enumerate(zip(blocks, layers_specs)):
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


def report_weight_distribution(
    run,
    frequencies=None,
    norms=None,
    scatter=None,
    scatter_layer=None,
):
    res = []
    if scatter:
        fig = Figure(fontsize=17)
        fig.plot(
            scatter,
            f"Motifs per layer for S={run[1]}, capacity={run[2]}",
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
            f"Motifs frequencies for S={run[1]}, capacity={run[2]}",
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
            f"Frequencies of Motifs' norms per layer for S={run[1]}, capacity={run[2]}",
            x_label="Norms",
            y_label="Frequency",
            logy=True,
            axis=None,
            symbol="x",
        )
        res.append(fig)

    if scatter_layer:
        fig = Figure(
            f"Frequencies of Motifs' norms per layer for S={run[1]}, capacity={run[2]}",
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


def plot_weights(run):
    model = load_model(run[0])

    blocks, layers_specs = extract_blocks(model, run[1])

    all_blocks = torch.cat(blocks, dim=0)

    if all_blocks.shape[1] == 2:
        _, counts = all_blocks.unique(dim=0, return_counts=True)

        sort_idx = torch.argsort(counts, descending=True)
        frequencies = {
            "All layers": (range(len(sort_idx)), counts[sort_idx] / len(counts))
        }

        # frequencies = frequency_data(all_blocks, blocks)
        # print(len(scatter_layer.items()))

        modules = module_data(all_blocks, blocks)
        scatter, scatter_layer = scatter_data(blocks, layers_specs, run[2])

        return report_weight_distribution(
            run, frequencies, modules, scatter, scatter_layer
        )

    elif all_blocks.shape[1] > 2:
        pca = PCA(n_components=0.9999)  # Retain all blocks
        pca.fit(all_blocks)

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


def plot_blocks(run, blocks, layers_specs):
    scatter, scatter_layer = scatter_data_numpy(blocks, layers_specs, run[2])
    return report_weight_distribution(run, scatter=scatter, scatter_layer=scatter_layer)


def scatter_data_numpy(blocks, layers_specs, capacity):
    scatter_per_layer = {}
    scatter_motifs_per_layer = {}

    for i, (b, s) in enumerate(zip(blocks, layers_specs)):
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
