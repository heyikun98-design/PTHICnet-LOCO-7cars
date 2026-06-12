# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_utils import PointNetSetAbstraction
from losses import KDEWeightedHuberLoss


class get_model(nn.Module):
    """
    E1 clean ablation model:
    - input: fused_input [B, 19, N]
    - remove thickness/material separate branches and cross-attention
    - keep PointNet++ backbone for isolated early-fusion gain
    """

    def __init__(
        self,
        normal_channel=False,
        num_point=102,
        age_embed_dim=8,
        shape_audit=False,
    ):
        super().__init__()
        self.normal_channel = normal_channel
        self.num_point = num_point
        self.shape_audit = shape_audit
        self._audit_printed = False

        # sample_and_group will concatenate grouped xyz (3) + points(D)
        # fused input carries D=16 (thickness1 + material15), so sa1 in_channel must be 19.
        sa1_in_channel = 22 if normal_channel else 19
        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=sa1_in_channel,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, 1024],
            group_all=True,
        )

        self.key_point_fc = nn.Linear(3, 64)
        self.age_group_embedding = nn.Embedding(num_embeddings=2, embedding_dim=age_embed_dim)
        self.regressor = nn.Sequential(
            nn.Linear(1024 + 64 + age_embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def _shape_audit(self, fused_input, xyz, points, l1_xyz, l1_points, l2_xyz, l2_points, l3_points):
        if self._audit_printed or not self.shape_audit:
            return
        self._audit_printed = True
        print("[E1 shape audit] fused_input:", tuple(fused_input.shape))
        print("[E1 shape audit] xyz:", tuple(xyz.shape), "points:", tuple(points.shape))
        print("[E1 shape audit] sa1.in_channel expected:", self.sa1.mlp_convs[0].in_channels)
        print("[E1 shape audit] l1_xyz:", tuple(l1_xyz.shape), "l1_points:", tuple(l1_points.shape))
        print("[E1 shape audit] sa2.in_channel expected:", self.sa2.mlp_convs[0].in_channels)
        print("[E1 shape audit] l2_xyz:", tuple(l2_xyz.shape), "l2_points:", tuple(l2_points.shape))
        print("[E1 shape audit] l3_points:", tuple(l3_points.shape))

    def forward(self, fused_input, hic_point, category, age_group):
        # fused_input [B, 19, N]
        xyz = fused_input[:, :3, :]
        points = fused_input[:, 3:, :]
        if self.normal_channel:
            # Optional normals mode assumes [xyz(3)+normal(3)+thickness(1)+mat(15)]
            points = fused_input[:, 3:, :]

        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)
        x_global = l3_points.view(fused_input.shape[0], 1024)

        self._shape_audit(fused_input, xyz, points, l1_xyz, l1_points, l2_xyz, l2_points, l3_points)

        hic_feat = self.key_point_fc(hic_point)
        age_feat = self.age_group_embedding(torch.clamp(age_group.long(), 0, 1))
        x_cat = torch.cat([x_global, hic_feat, age_feat], dim=1)
        pred = self.regressor(x_cat)
        return pred, l3_points


class get_loss(KDEWeightedHuberLoss):
    def __init__(self, kde_reference_csv=None, eps=1e-6, delta=5):
        super().__init__(kde_reference_csv=kde_reference_csv, eps=eps, delta=delta)

