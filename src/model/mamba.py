import torch
import torch.nn as nn
from transformers import Mamba2Config, Mamba2Model


class Mamba(nn.Module):
    def __init__(
        self,
        n_motifs,
        num_heads,
        head_dim,
        hidden_size,
        num_rows,
        num_cols,
        num_layers,
        state_size,
        num_hidden_layers,
        n_groups,
        hidden_act,
        time_step_min,
        time_step_max,
        out_channels,
        output_layers,
        mamba_chunk_size=None,
    ):
        super().__init__()
        mamba_kwargs = dict(
            output_hidden_states=False,
            num_heads=num_heads,
            head_dim=head_dim,
            vocab_size=0,
            hidden_size=hidden_size,
            state_size=state_size,
            num_hidden_layers=num_hidden_layers,
            n_groups=n_groups,
            hidden_act=hidden_act,
            time_step_min=time_step_min,
            time_step_max=time_step_max,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        # Mamba2's torch_forward materializes [B, H, n_chunks, C, C, head_dim]
        # tensors in the SSD diagonal term — C = ``chunk_size`` is the dominant
        # squared factor, so lower it for long input sequences.
        if mamba_chunk_size is not None:
            mamba_kwargs["chunk_size"] = int(mamba_chunk_size)
        config = Mamba2Config(**mamba_kwargs)

        self._hidden_size = hidden_size

        self.motif_embedding = nn.Embedding(
            num_embeddings=n_motifs, embedding_dim=hidden_size
        )
        self.row_embedding = nn.Embedding(
            num_embeddings=num_rows, embedding_dim=hidden_size
        )
        self.col_embedding = nn.Embedding(
            num_embeddings=num_cols, embedding_dim=hidden_size
        )
        self.layer_embedding = nn.Embedding(
            num_embeddings=num_layers, embedding_dim=hidden_size
        )

        self.model = Mamba2Model(config)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(output_layers)]
        )
        self.output_head = nn.Linear(hidden_size, out_channels)
        self.relu = nn.ReLU(inplace=True)

        import transformers.models.mamba2.modeling_mamba2 as _m2

        _m2.is_fast_path_available = False

    def forward(self, motif, layer, n_rows, n_cols, row=0, col=0):
        device = self.row_embedding.weight.device
        squeeze_batch = motif.dim() == 0
        if squeeze_batch:
            motif = motif.unsqueeze(0)

        motif_em = self.motif_embedding(motif)
        layer_em = self.layer_embedding(layer)
        rows_em = self.row_embedding(torch.arange(row, row + n_rows, device=device))
        cols_em = self.col_embedding(torch.arange(col, col + n_cols, device=device))
        grid_embeddings = rows_em.unsqueeze(1) + cols_em.unsqueeze(0)

        B = motif_em.shape[0]
        prefix = (motif_em + layer_em.unsqueeze(0)).unsqueeze(1)
        grid = (
            grid_embeddings.view(-1, self._hidden_size).unsqueeze(0).expand(B, -1, -1)
        )
        x = torch.cat([prefix, grid], dim=1)

        out = self.model(inputs_embeds=x)[0][:, 1:]
        for linear in self.layers:
            out = nn.ReLU(linear(out))
        out = self.output_head(out)
        if squeeze_batch:
            out = out.squeeze(0)
        return out
