"""
边界框工具模块

提供边界框特征提取和处理的统一工具函数。
"""

import torch
from torch import Tensor
from typing import Optional


def extract_bbox_features(
    bbox_data: Tensor,
    num_classes: int = 80
) -> Tensor:
    """
    从边界框数据中提取特征 [cx, cy, obj_type]
    
    Args:
        bbox_data: 边界框数据，形状为 [B, 7] [x1, y1, x2, y2, label, cx, cy]
                   其中 cx, cy 是归一化的中心点坐标
        num_classes: 物体类别总数，用于归一化物体类型，默认为80（COCO数据集）
    
    Returns:
        bbox_features: [B, 3] 边界框特征 [cx, cy, obj_type_norm]
                       - cx: 归一化的中心点x坐标
                       - cy: 归一化的中心点y坐标  
                       - obj_type_norm: 归一化的物体类型 (0-1范围)
    
    Example:
        >>> bbox_data = torch.tensor([[10, 20, 50, 80, 1, 0.3, 0.5],
        ...                           [15, 25, 60, 90, 2, 0.4, 0.6]])
        >>> features = extract_bbox_features(bbox_data)
        >>> features.shape
        torch.Size([2, 3])
    """
    if bbox_data.dim() == 2 and bbox_data.size(1) >= 7:
        # 提取归一化的中心点坐标和物体类型
        cx_norm: Tensor = bbox_data[:, 5]  # 第6列是归一化的cx
        cy_norm: Tensor = bbox_data[:, 6]  # 第7列是归一化的cy
        obj_type: Tensor = bbox_data[:, 4]  # 第5列是物体类型

        # 将物体类型归一化到 [0, 1] 范围
        obj_type_norm: Tensor = obj_type / float(num_classes)

        bbox_features: Tensor = torch.stack([cx_norm, cy_norm, obj_type_norm], dim=1)
    else:
        # 如果没有边界框数据，使用默认值
        batch_size: int = bbox_data.size(0) if bbox_data.dim() > 0 else 1
        bbox_features = torch.zeros(batch_size, 3, dtype=bbox_data.dtype, device=bbox_data.device)

    return bbox_features


def validate_bbox_data(bbox_data: Tensor) -> bool:
    """
    验证边界框数据格式是否正确
    
    Args:
        bbox_data: 边界框数据张量
        
    Returns:
        bool: 如果数据格式正确返回True，否则返回False
    """
    if not isinstance(bbox_data, Tensor):
        return False
    if bbox_data.dim() != 2:
        return False
    if bbox_data.size(1) < 7:
        return False
    return True


def get_bbox_center(
    bbox_data: Tensor,
    normalized: bool = True
) -> Tensor:
    """
    获取边界框的中心点坐标
    
    Args:
        bbox_data: 边界框数据，形状为 [B, 7] 或 [B, 4+]
        normalized: 是否返回归一化坐标（如果bbox_data包含归一化坐标）
        
    Returns:
        center: [B, 2] 中心点坐标 [cx, cy]
    """
    if normalized and bbox_data.size(1) >= 7:
        # 使用预计算的归一化中心点
        return bbox_data[:, 5:7]
    else:
        # 从边界框坐标计算中心点
        x1, y1, x2, y2 = bbox_data[:, 0], bbox_data[:, 1], bbox_data[:, 2], bbox_data[:, 3]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return torch.stack([cx, cy], dim=1)


def get_bbox_size(bbox_data: Tensor) -> Tensor:
    """
    获取边界框的尺寸
    
    Args:
        bbox_data: 边界框数据，形状为 [B, 4+] [x1, y1, x2, y2, ...]
        
    Returns:
        size: [B, 2] 边界框尺寸 [width, height]
    """
    x1, y1, x2, y2 = bbox_data[:, 0], bbox_data[:, 1], bbox_data[:, 2], bbox_data[:, 3]
    width = x2 - x1
    height = y2 - y1
    return torch.stack([width, height], dim=1)
