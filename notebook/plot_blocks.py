# %% [markdown]
#
# %%
from tqdm import trange
import numpy as np
from src.view.weight_distribution import load_model, extract_blocks_2d
import torch
from src.quantizers.momos import _nearest_motifs_chunked
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages

# %%
rows, cols = 1, 2
checkpoint = "artifacts/model-sh4p5gxq:v0/model.ckpt"
filename = "masks_r1_c2_cap0.001.pdf"
figures = []
# %%
model = load_model(checkpoint)
blocks, layers_specs = extract_blocks_2d(model, rows, cols)
all_blocks = torch.cat(blocks, dim=0)
# %%
for i, specs in enumerate(layers_specs):
    shape = specs[-1]
    print(f"\item layer {i + 1}: {shape[0] * shape[1] // 2 // 4096}")
# %%
motifs = all_blocks.unique(dim=0)
M = _nearest_motifs_chunked(all_blocks, motifs, 256)
# %%
shape = layers_specs[0][-1]
shape = (shape[0] // rows, shape[1] // cols)
M2d = torch.reshape(M[: shape[0] * shape[1]], shape)
# %%
M2d_np = M2d.cpu().numpy()
n_motifs = int(motifs.shape[0])
brown_cmap = LinearSegmentedColormap.from_list(
    "black_brown_white",
    [(0.0, 0.0, 0.0), (0.55, 0.27, 0.07), (1.0, 1.0, 1.0)],
    N=n_motifs,
)
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(
    M2d_np, cmap=brown_cmap, vmin=0, vmax=n_motifs - 1, interpolation="nearest"
)
fig.colorbar(im, ax=ax, label="motif index")
ax.set_title(f"Motif assignment ({n_motifs} unique)")
figures.append(fig)
# %%
for i in trange(n_motifs):
    M2d_mask_np = (M2d != i).cpu().numpy()
    n_motifs = 2
    brown_cmap = LinearSegmentedColormap.from_list(
        "black_brown_white",
        [(0.0, 0.0, 0.0), (0.55, 0.27, 0.07), (1.0, 1.0, 1.0)],
        N=2,
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        M2d_mask_np, cmap=brown_cmap, vmin=0, vmax=1, interpolation="nearest"
    )
    fig.colorbar(im, ax=ax, label="motif index")
    ax.set_title(f"Motif {i}")
    figures.append(fig)
# %%
with PdfPages("assets/" + filename) as pdf:
    for fig in figures:
        pdf.savefig(fig, bbox_inches="tight")
# %%
for i in layers_specs:
    rows, cols = i[-1]
    print(rows * cols / 4096)
# M2d.shape[0] * M2d.shape[1] / 4096
