"""
HF2-VAD 数据集模块

提供原始数据集和优化数据集的实现。
"""

from datasets.dataset import (
    Chunked_sample_dataset,
    get_dataset,
    get_foreground,
    img_batch_tensor2numpy,
    img_tensor2numpy,
    bbox_collate,
    extract_bbox_features_from_data,
)

from datasets.dataset_optimized import (
    BatchImageProcessor,
    MemoryMappedChunkedDataset,
    SingleChunkMemoryMappedDataset,
    ChunkedIterableDataset,
    get_optimized_dataloader,
    AutoNumWorkersConfig,
    get_auto_num_workers,
)

__all__ = [
    # 原始数据集
    'Chunked_sample_dataset',
    'get_dataset',
    'get_foreground',
    'img_batch_tensor2numpy',
    'img_tensor2numpy',
    'bbox_collate',
    'extract_bbox_features_from_data',
    # 优化数据集
    'BatchImageProcessor',
    'MemoryMappedChunkedDataset',
    'SingleChunkMemoryMappedDataset',
    'ChunkedIterableDataset',
    'get_optimized_dataloader',
    # 自动配置
    'AutoNumWorkersConfig',
    'get_auto_num_workers',
]
