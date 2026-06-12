# -*- coding: utf-8*-
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction
import torch
import os
from pathlib import Path
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads=4, dropout=0.1):
        super().__init__()
        assert dim_out % num_heads == 0
        self.dim_head = dim_out // num_heads
        self.num_heads = num_heads

        self.q_proj = nn.Linear(dim_in, dim_out)
        self.k_proj = nn.Linear(dim_in, dim_out)
        self.v_proj = nn.Linear(dim_in, dim_out)
        self.out_proj = nn.Linear(dim_out, dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, N, C]
        B, N, _ = x.shape
        Q = self.q_proj(x).view(B, N, self.num_heads, self.dim_head).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.num_heads, self.dim_head).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.num_heads, self.dim_head).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / (self.dim_head ** 0.5)  # [B, H, N, N]
        attn = self.dropout(attn.softmax(dim=-1))

        out = attn @ V  # [B, H, N, d]
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.out_proj(out)  # [B, N, dim_out]

class FeatureSelfAttentionSameDim(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.attn = MultiHeadSelfAttention(embed_dim, embed_dim, num_heads)
        self.output_proj = nn.Linear(embed_dim, 1)

    def forward(self, x):  # x: [B, N]
        B, N = x.shape
        x = x.unsqueeze(-1)        # [B, N, 1]
        x_embed = self.embed(x)    # [B, N, embed_dim]
        x_attn = self.attn(x_embed) # [B, N, embed_dim]
        out = self.output_proj(x_attn) # [B, N, 1]
        return out.squeeze(-1)       # [B, N]




class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, dim_out, num_heads=4):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.query_proj = nn.Linear(dim_q, dim_out)
        self.key_proj = nn.Linear(dim_kv, dim_out)
        self.value_proj = nn.Linear(dim_kv, dim_out)
        self.softmax = nn.Softmax(dim=-1)
        self.out_proj = nn.Linear(dim_out, dim_out)

    def forward(self, q, kv):
        # q: [B, dim_q], kv: [B, dim_kv]
        q = self.query_proj(q).unsqueeze(1)  # [B, 1, dim_out]
        k = self.key_proj(kv).unsqueeze(1)   # [B, 1, dim_out]
        v = self.value_proj(kv).unsqueeze(1) # [B, 1, dim_out]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)  # [B, 1, 1]
        attn_weights = self.softmax(attn_scores)  # [B, 1, 1]
        attended = torch.matmul(attn_weights, v)  # [B, 1, dim_out]

        return self.out_proj(attended.squeeze(1))  # [B, dim_out]

class get_model(nn.Module):
    def __init__(self, normal_channel=True, num_point=102):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3

        self.normal_channel = normal_channel
        self.num_point = num_point
        # PointNet++ 分层特征提取（隐藏层）
        self.mlp1_dims = [64, 64, 128]
        self.mlp2_dims = [128, 128, 256]
        self.mlp3_dims = [256, 512, 1024]
        self.key_point_dim = 64
        self.thickness_dim = 32
        # self.part_feat_dim = 256  # 移除part_name相关维度
        self.mat_feat_dim = 256
        self.age_feat_dim = 8


        self.hidden1_dim = 512
        self.hidden2_dim = 256
        self.final_dim = 1

        self.cross_attn_out = 1024
        self.self_att_use1=False
        self.self_att_use2=True
        # PointNet++ 分层特征提取（隐藏层）
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                          mlp=[256, 512, 1024], group_all=True)

        # 新增处理关键点和类别的层
        self.key_point_fc = nn.Linear(3, 64)  # 处理关键点坐标，HIC坐标点的位置
        # self.thickness_fc = nn.Linear(self.num_point, 512)  # 处理关键点坐标，HIC坐标点的位置
        # self.category_embedding = nn.Embedding(num_embeddings=5, embedding_dim=64)  # 假设有num_embeddings个类别
        # self.part_name_embedding = nn.Embedding(num_embeddings=17, embedding_dim=256)  # 移除part_name处理
        
        # 替换材料嵌入层为材料物理属性处理层
        # MaterialProps包含: density, youngs_modulus, poisson_ratio, stress_strain_curve1(6个), stress_strain_curve2(6个)
        # 总共 1 + 1 + 1 + 6 + 6 = 15 个物理属性
        self.material_props_fc = nn.Linear(15, 256)  # 将15维材料物理属性映射到256
        self.material_props_bn = nn.BatchNorm1d(256)
        self.material_props_drop = nn.Dropout(0.1)
        
        self.age_group_embedding = nn.Embedding(num_embeddings=2, embedding_dim=8)  # 头型信息，假设有num_embeddings个类别
        # self.conv1 = nn.Conv2d(self.num_point, 512,  kernel_size=(5, 1), padding=(2, 0))
        self.maxpool = nn.MaxPool2d(kernel_size=(512, 1))
        self.flatten = nn.Flatten()

        # 添加thickness降维层
        self.thickness_fc = nn.Linear(self.num_point, 32)  # 将102维降到32维
        self.thickness_bn = nn.BatchNorm1d(32)
        self.thickness_drop = nn.Dropout(0.1)

        # 1024：点云特征
        # 64：关键点特征
        # 102：厚度特征
        # 8：年龄组特征
        # 102：零部件和材料特征的组合
        # 然后通过fc1层映射到512维

        # 扩展全连接层以接收额外信息
        self.fc1 = nn.Linear(self.cross_attn_out + self.cross_attn_out, self.hidden1_dim )  # 修改输入维度，thickness从102降到32
        self.bn1 = nn.BatchNorm1d(self.hidden1_dim )
        self.drop1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(self.hidden1_dim , self.hidden2_dim )
        self.bn2 = nn.BatchNorm1d(self.hidden2_dim )
        self.drop2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(self.hidden2_dim , 1)  # 分类变成回归

        self.q_dim=self.mlp3_dims[-1]
        self.kv_dim=64 + 32 + 8 + 256  # key_point_dim + thickness_dim + age_feat_dim + mat_feat_dim (移除part_feat_dim)
        self.cross_attn_x_to_x2 = CrossAttention(dim_q=self.q_dim, dim_kv=self.kv_dim, dim_out=self.cross_attn_out)
        self.cross_attn_x2_to_x = CrossAttention(dim_q=self.kv_dim, dim_kv=self.q_dim, dim_out=self.cross_attn_out)

        self.self_attn1= FeatureSelfAttentionSameDim(embed_dim=64, num_heads=2)
        self.self_attn2= FeatureSelfAttentionSameDim(embed_dim=64, num_heads=2)


    def forward(self, xyz, key_point, category, thickness, material_props, age_group):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        # 处理点云数据
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)

        # 处理关键点和类别信息
        # 确保 thickness 形状正确：[B, num_point] 或 [B, num_point, 1]
        if thickness.dim() == 3:
            thickness = torch.squeeze(thickness, dim=2)  # [B, N, 1] -> [B, N]
        elif thickness.dim() == 2:
            pass  # 已经是 [B, N]
        else:
            raise ValueError(f"Unexpected thickness shape: {thickness.shape}, expected [B, N] or [B, N, 1]")
        
        # 检查维度是否匹配
        B_thick, N_thick = thickness.shape
        if N_thick != self.num_point:
            # 如果维度不匹配，进行插值或填充
            if N_thick < self.num_point:
                # 使用插值扩展到 num_point
                thickness = F.interpolate(
                    thickness.unsqueeze(1), 
                    size=self.num_point, 
                    mode='linear', 
                    align_corners=False
                ).squeeze(1)
            else:
                # 如果超过，则截断或下采样
                thickness = F.interpolate(
                    thickness.unsqueeze(1), 
                    size=self.num_point, 
                    mode='linear', 
                    align_corners=False
                ).squeeze(1)
        
        thickness = thickness.float()  # 确保thickness是float32类型
        thickness = self.thickness_drop(F.relu(self.thickness_bn(self.thickness_fc(thickness))))  # [B, 32]
        key_point_feat = self.key_point_fc(key_point)  # [B, 64]
        
        # 处理材料物理属性
        # material_props: [B, num_point, 15] 包含所有材料物理属性
        material_props = material_props.float()  # 确保是float类型
        
        # 对每个点的材料属性进行处理
        B, N, _ = material_props.shape
        material_props_reshaped = material_props.view(B * N, 15)  # [B*N, 15]
        material_props_processed = self.material_props_drop(
            F.relu(self.material_props_bn(self.material_props_fc(material_props_reshaped)))
        )  # [B*N, 256]
        material_props_processed = material_props_processed.view(B, N, 256)  # [B, N, 256]
        
        # 确保 age_group 索引在有效范围内 [0, 1]
        age_group = torch.clamp(age_group.long(), 0, 1)
        age_feat = self.age_group_embedding(age_group)  # [B, 8]

        # 特征融合 (移除part_feat)
        # x2现在只包含material_props_processed
        x2 = material_props_processed  # [B, N, 256]
        
        # 对x2进行全局平均池化，然后与key_point_feat, thickness, age_feat拼接
        x2 = torch.mean(x2, dim=1)  # [B, 256] - 对点维度进行平均池化
        
        x2 = torch.cat([key_point_feat, thickness, age_feat, x2], dim=1)  # [B, 64+32+8+256=360]

        if self.self_att_use1:
            x = self.self_attn1(x)  # [B, 512]
        if self.self_att_use2:
            x2 = self.self_attn2(x2)  # [B, 512]



        x_attn = self.cross_attn_x_to_x2(q=x, kv=x2)  # [B, 512]
        x2_attn = self.cross_attn_x2_to_x(q=x2, kv=x)  # [B, 512]

        # 融合两路注意力结果
        x_fused = torch.cat([x_attn, x2_attn], dim=1)  # [B, 1024]



        # 全连接层处理
        x = self.drop1(F.relu(self.bn1(self.fc1(x_fused))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x, l3_points



# 定义权重函数

class get_loss(nn.Module):
    def __init__(self, eps=1e-6, delta=5, kde_reference_csv=None):
        super().__init__()
        self.eps = eps
        self.delta = float(delta)
        self.kde_model = None
        self.min = 0.0
        self.max = 1.0

        if kde_reference_csv and os.path.exists(kde_reference_csv):
            y_train = pd.read_csv(kde_reference_csv).to_numpy().flatten()
            if y_train.size >= 2:
                kde = gaussian_kde(y_train)
                dens = kde(y_train)
                self.kde_model = kde
                self.min = float(dens.min())
                self.max = float(dens.max())
        print(f"[LossConfig] delta={self.delta:.6g} kde={'enabled' if self.kde_model is not None else 'disabled'}")

    def _get_kde_weight(self, y_vals):
        if self.kde_model is None:
            return np.ones((y_vals.shape[0],), dtype=np.float32)
        dens = self.kde_model(y_vals[:, 0])
        dens = (dens - self.min) / (self.max - self.min + self.eps)
        dens = 1.0 - dens
        dens = np.maximum(dens, self.eps)
        return (dens / np.mean(dens)).astype(np.float32)

    def forward(self, pred, target, trans_feat=None):
        target_np = target.detach().cpu().numpy()
        weights_np = self._get_kde_weight(target_np)
        weights = torch.tensor(weights_np, dtype=torch.float32, device=pred.device).view(-1, 1)

        diff = pred - target
        abs_diff = torch.abs(diff)
        mse_loss = 0.5 * (diff ** 2)
        mae_loss = self.delta * (abs_diff - 0.5 * self.delta)
        loss = torch.where(abs_diff <= self.delta, mse_loss, torch.abs(mae_loss))
        return torch.mean(loss * weights)

#
# class get_loss(nn.Module):
#     def __init__(self):
#         super(get_loss, self).__init__()
#
#     def forward(self, pred, target, trans_feat):
#         # total_loss = F.mse_loss(pred, target)
#
#         total_loss = F.smooth_l1_loss(pred, target)
#         return total_loss
