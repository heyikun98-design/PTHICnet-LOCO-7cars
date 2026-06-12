import torch
import torch.nn as nn

from models.point_transformer_block import PointTransformerBlock
from models.point_transformer_utils import (
    farthest_point_sample,
    index_points,
    query_ball_point,
    square_distance,
)


class TransitionDown(nn.Module):
    def __init__(self, npoint, radius, nsample, in_dim, out_dim):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.block = PointTransformerBlock(in_dim)
        self.proj = nn.Sequential(nn.Linear(in_dim * 3, out_dim), nn.ReLU())
        self._diagnosed = False

    def forward(self, xyz, feats):
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)
        group_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, group_idx)
        grouped_feats = index_points(feats, group_idx)

        if not self._diagnosed:
            self._diagnosed = True
            with torch.no_grad():
                sqrdists = square_distance(new_xyz, xyz)
                within = (sqrdists <= self.radius ** 2).sum(dim=-1).float()
                avg = within.mean().item()
                pct = (within >= self.nsample).float().mean().item() * 100
                print(
                    f"[BallQuery] npoint={self.npoint} r={self.radius:.0f} "
                    f"nsample={self.nsample} | avg_valid={avg:.1f}  "
                    f"pct_full(>={self.nsample})={pct:.0f}%"
                )

        rel_pos = new_xyz.unsqueeze(2) - grouped_xyz
        center_feats = index_points(feats, fps_idx)
        out = self.block(center_feats, grouped_feats, rel_pos)
        max_pool = torch.max(grouped_feats, dim=2)[0]
        avg_pool = torch.mean(grouped_feats, dim=2)
        pooled = torch.cat([max_pool, avg_pool], dim=-1)
        out = self.proj(torch.cat([out, pooled], dim=-1))
        return new_xyz, out

