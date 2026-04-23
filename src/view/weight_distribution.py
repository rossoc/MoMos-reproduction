import torch

from src.view.figure import Figure
from src.quantizers import iter_trainable_params, tensor_to_blocks
from src.model import MLP

def extract_blocks(model, block_size):
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


def compute_per_layer_metrics(blocks, layers_specs, capacity):
    scatter_per_layer = {}
    scatter_motifs_per_layer = {}

    for i, (b, s) in enumerate(zip(blocks, layers_specs)):
        uniques_motifs_l = b.unique(dim=0)
        scatter_per_layer[f"Layer {i + 1}"] = uniques_motifs_l.T
        if i % 2 == 0 and i < 7:
            scatter_motifs_per_layer[f"Layer {i + 1} with n_params={capacity}"] = {
                "scatter": uniques_motifs_l.T
            }

    return (
        scatter_per_layer,
        scatter_motifs_per_layer,
    )


def report_weight_distribution(
    run, sorted_counts, scatter_per_layer, scatter_per_layer_single
):
    fig_scatter_per_layer = Figure()
    fig_scatter_per_layer.plot(
        scatter_per_layer,
        f"Motifs per layer for S={run[1]}, capacity={run[2]}",
        symbol="o",
        axis=None,
        x_label="$X_1$",
        y_label="$X_2$",
    )

    fig_counts = Figure()
    fig_counts.plot(
        {"Counts": (range(len(sorted_counts)), sorted_counts / len(sorted_counts))},  # type: ignore
        f"Overall counts for S={run[1]}, capacity={run[2]}",
        x_label="Motifs",
        y_label="Frequency",
    )

    fig_scatter_per_single_layer = Figure(
        f"Motifs per layer for S={run[1]}, capacity={run[2]}", ncols=2, nrows=2
    )

    for (n, d), c in zip(
        scatter_per_layer_single.items(), ["red", "green", "pink", "coral"]
    ):
        _ = fig_scatter_per_single_layer.plot(
            d,
            n,
            legend=False,
            symbol="o",
            axis=None,
            x_label="$X_1$",
            y_label="$X_2$",
            colors=[c],
        )

    return [
        fig_scatter_per_layer,
        fig_scatter_per_single_layer,
        fig_counts,
    ]


def plot_weights(run):
    model = load_model(run[0])

    blocks, layers_specs = extract_blocks(model, run[1])

    all_blocks = torch.cat(blocks, dim=0)
    motifs, counts = all_blocks.unique(dim=0, return_counts=True)

    sort_idx = torch.argsort(counts, descending=True)

    scatter_per_layer, scatter_per_single_layer = compute_per_layer_metrics(
        blocks, layers_specs, run[2]
    )

    return report_weight_distribution(
        run, counts[sort_idx], scatter_per_layer, scatter_per_single_layer
    )
