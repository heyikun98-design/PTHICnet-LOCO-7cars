import torch
import torch.nn as nn

from models.film import FiLM
from models.point_transformer_layer import TransitionDown


class PT_HICnet(nn.Module):
    def __init__(
        self,
        in_channels=19,
        use_normals=False,
        pt_npoints=(512, 128, 32, 1),
        pt_radius=(60, 150, 400, 1500),
        pt_nsample=(32, 32, 32, 32),
        film_mode="none",
        age_embed_dim=16,
    ):
        super().__init__()
        self.use_normals = use_normals
        self.expected_channels = 22 if use_normals else in_channels
        c0 = 64
        self.input_proj = nn.Sequential(
            nn.Conv1d(self.expected_channels, c0, 1),
            nn.BatchNorm1d(c0),
            nn.ReLU(),
        )

        dims = [c0, 128, 256, 512, 1024]
        self.down1 = TransitionDown(pt_npoints[0], pt_radius[0], pt_nsample[0], dims[0], dims[1])
        self.down2 = TransitionDown(pt_npoints[1], pt_radius[1], pt_nsample[1], dims[1], dims[2])
        self.down3 = TransitionDown(pt_npoints[2], pt_radius[2], pt_nsample[2], dims[2], dims[3])
        self.down4 = TransitionDown(pt_npoints[3], pt_radius[3], pt_nsample[3], dims[3], dims[4])

        self.hic_point_fc = nn.Linear(3, 64)
        self.age_embedding = nn.Embedding(2, age_embed_dim)
        self.film_mode = film_mode
        self.film = FiLM(age_embed_dim, dims[4]) if film_mode == "global" else None
        self.deep_films = (
            nn.ModuleList(
                [
                    FiLM(age_embed_dim, dims[1]),
                    FiLM(age_embed_dim, dims[2]),
                    FiLM(age_embed_dim, dims[3]),
                    FiLM(age_embed_dim, dims[4]),
                ]
            )
            if film_mode == "deep"
            else None
        )
        self.regressor_head = nn.Sequential(
            nn.Linear(dims[4] + 64, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
    def forward(self, fused_input, hic_point, age_group):
        # fused_input: [B, C, N]
        if fused_input.shape[1] != self.expected_channels:
            raise ValueError(
                f"Expected {self.expected_channels} channels, got {fused_input.shape[1]}."
            )
        xyz = fused_input[:, :3, :].transpose(1, 2).contiguous()
        feats = self.input_proj(fused_input).transpose(1, 2).contiguous()

        age_emb = self.age_embedding(torch.clamp(age_group.long(), 0, 1))
        xyz, feats = self.down1(xyz, feats)
        if self.deep_films is not None:
            feats = self._apply_deep_film(self.deep_films[0], feats, age_emb)
        xyz, feats = self.down2(xyz, feats)
        if self.deep_films is not None:
            feats = self._apply_deep_film(self.deep_films[1], feats, age_emb)
        xyz, feats = self.down3(xyz, feats)
        if self.deep_films is not None:
            feats = self._apply_deep_film(self.deep_films[2], feats, age_emb)
        xyz, feats = self.down4(xyz, feats)
        if self.deep_films is not None:
            feats = self._apply_deep_film(self.deep_films[3], feats, age_emb)

        x_global = feats.squeeze(1)
        if self.film is not None:
            x_global = self.film(x_global, age_emb)
        hic_feat = self.hic_point_fc(hic_point)
        x_cat = torch.cat([x_global, hic_feat], dim=1)
        pred = self.regressor_head(x_cat)
        return pred, feats.transpose(1, 2).contiguous()

    @staticmethod
    def _apply_deep_film(film_layer, feats, age_emb):
        bsz, npoint, dim = feats.shape
        cond = age_emb.unsqueeze(1).repeat(1, npoint, 1).reshape(bsz * npoint, -1)
        flat = feats.reshape(bsz * npoint, dim)
        mod = film_layer(flat, cond)
        return mod.reshape(bsz, npoint, dim)

