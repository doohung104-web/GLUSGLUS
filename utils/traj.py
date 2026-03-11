import torch
import numpy as np

def extract_scaled_centroid(mask, target_range=1000):
    """
    计算二值掩码的质心并缩放至 
    mask: torch.Tensor 或 numpy.array, 形状为 (H, W)，值为 0 或 1
    """
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    if not isinstance(mask, torch.Tensor):
        return ""

    if mask.ndim < 2:
        return ""

    # 统一为二维空间掩码，兼容 [1,H,W] / [N,H,W] 等输入
    if mask.ndim > 2:
        h, w = mask.shape[-2], mask.shape[-1]
        mask = mask.reshape(-1, h, w).bool().any(dim=0)
    else:
        mask = mask.bool()
    
    # 找到所有非零像素的索引
    coords = torch.nonzero(mask)
    if len(coords) == 0:
        # 如果掩码为空（物体消失），返回特殊标记或中心点
        return "" 

    # 计算质心 (y, x)
    y_mean, x_mean = coords.float().mean(dim=0)
    
    # 获取图像尺寸
    h, w = mask.shape

    # 缩放至  整数区间
    x_scaled = int((x_mean / w) * target_range)
    y_scaled = int((y_mean / h) * target_range)
    
    # 限制范围在 0-1000 之间
    x_scaled = max(0, min(target_range, x_scaled))
    y_scaled = max(0, min(target_range, y_scaled))
    
    # 格式化为文本
    return f"[{x_scaled}, {y_scaled}]"


