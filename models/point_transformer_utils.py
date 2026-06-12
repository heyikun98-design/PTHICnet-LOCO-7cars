import torch


def square_distance(src, dst):
    bsz, n, _ = src.shape
    _, m, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(bsz, n, 1)
    dist += torch.sum(dst ** 2, -1).view(bsz, 1, m)
    return dist


def index_points(points, idx):
    device = points.device
    bsz, n, _ = points.shape
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(bsz, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    )
    idx = torch.clamp(idx, 0, n - 1)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    bsz, n, _ = xyz.shape
    if n < npoint:
        return torch.randint(0, n, (bsz, npoint), dtype=torch.long, device=device)
    centroids = torch.zeros(bsz, npoint, dtype=torch.long, device=device)
    distance = torch.ones(bsz, n, device=device) * 1e10
    farthest = torch.randint(0, n, (bsz,), dtype=torch.long, device=device)
    batch_indices = torch.arange(bsz, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(bsz, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    bsz, n, _ = xyz.shape
    _, s, _ = new_xyz.shape
    group_idx = torch.arange(n, dtype=torch.long, device=device).view(1, 1, n).repeat([bsz, s, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = n
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(bsz, s, 1).repeat([1, 1, nsample])
    mask = group_idx == n
    group_idx[mask] = group_first[mask]
    return torch.clamp(group_idx, 0, n - 1)

