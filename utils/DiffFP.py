import torch

def get_diff_fp_compression_mask(pixel_values, grid_size, merge_size=2, threshold=0.1, min_tokens=1):
    """
    依照 VideoLLaMA 3 官方逻辑实现的 DiffFP 剪枝器
    
    Args:
        pixel_values: Tensor [Total_Patches, C_dim] 经过 Patchify 后的像素值
        grid_size: Tuple (T, H, W) 视频的帧数、高度(patch数)、宽度(patch数)
        merge_size: 空间下采样步长，VideoLLaMA 3 默认为 2
        threshold: 差分阈值，论文默认 0.1
        min_tokens: 每一帧至少保留的 Token 数量，防止全黑帧被完全删掉
    """
    t, h, w = grid_size
    # 1. 还原为帧序列结构 [T, Patches_per_frame, C_dim]
    # 注意：这里的 patches 数量是 (h // merge_size) * (w // merge_size)
    num_patches_per_frame = (h // merge_size) * (w // merge_size)
    images = pixel_values.view(t, num_patches_per_frame, -1)

    # 2. 计算相邻帧之间的像素差异 (Temporal Difference)
    # diff[t] = |frame[t] - frame[t-1]|
    pixel_diff = images[1:] - images[:-1]
    
    # 3. 计算每个 Patch 的平均 1-范数距离并归一化到 0-255 空间
    # VideoLLaMA 3 实现中乘以了 255 以对应像素值阈值
    pixel_diff = torch.abs(pixel_diff).mean(dim=-1) * 255 
    
    # 4. 第一帧总是保留 (设置一个超过阈值的初值)
    first_frame_delta = torch.full_like(pixel_diff[0:1], threshold + 1)
    pixel_diff = torch.cat([first_frame_delta, pixel_diff], dim=0)
    
    # 5. 生成压缩掩码：差异大于阈值的保留
    mask = pixel_diff > threshold
    
    # 6. 安全兜底：确保每一帧至少保留 min_tokens 个 Token
    # 如果某一帧所有 patch 变化都很小，强制保留前 min_tokens 个
    low_change_frames = torch.nonzero(mask.sum(dim=1) < min_tokens)[:, 0]
    mask[low_change_frames, :min_tokens] = True
    
    return mask.flatten() # 返回一维布尔掩码
