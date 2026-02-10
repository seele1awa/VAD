"""
优化的数据集模块

本模块实现了内存映射和批量处理优化的数据加载器，
用于提升 HF2-VAD 项目的训练效率和降低内存占用。

主要组件:
- MemoryMappedChunkedDataset: 使用内存映射的分块数据集
- BatchImageProcessor: 批量图像处理器
- ChunkedIterableDataset: 支持跨chunk无缝数据流的可迭代数据集
"""

import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torchvision.transforms as transforms
from typing import List, Tuple, Dict, Optional, Any, Iterator
from threading import Thread
from queue import Queue
import joblib
import tempfile
import shutil


# 默认变换
transform = transforms.Compose([
    transforms.ToTensor(),
])


class BatchImageProcessor:
    """
    批量图像处理器
    
    提供批量 resize 和 crop_and_resize 操作，
    相比逐个处理可显著提升效率。
    
    Attributes:
        interpolation: int - OpenCV 插值方法
    
    Methods:
        batch_resize: 批量调整图像大小
        batch_crop_and_resize: 批量裁剪并调整大小
    """
    
    def __init__(self, interpolation: int = cv2.INTER_LINEAR):
        """
        初始化批量图像处理器
        
        Args:
            interpolation: OpenCV 插值方法，默认为 cv2.INTER_LINEAR
        """
        self.interpolation = interpolation
    
    def batch_resize(
        self, 
        images: np.ndarray, 
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        批量调整图像大小
        
        Args:
            images: 输入图像数组，形状为 [N, H, W, C] 或 [N, C, H, W]
            target_size: 目标大小 (width, height)
        
        Returns:
            调整大小后的图像数组，保持输入形状格式
        """
        if images.ndim != 4:
            raise ValueError(f"Expected 4D array, got {images.ndim}D")
        
        n_images = images.shape[0]
        target_w, target_h = target_size
        
        # 检测输入格式 (NHWC 或 NCHW)
        is_channel_first = images.shape[1] <= 4 and images.shape[3] > 4
        
        if is_channel_first:
            # NCHW -> NHWC
            images = np.transpose(images, (0, 2, 3, 1))
        
        # 批量 resize
        resized = np.zeros((n_images, target_h, target_w, images.shape[3]), dtype=images.dtype)
        for i in range(n_images):
            resized[i] = cv2.resize(images[i], target_size, interpolation=self.interpolation)
        
        if is_channel_first:
            # NHWC -> NCHW
            resized = np.transpose(resized, (0, 3, 1, 2))
        
        return resized
    
    def batch_crop_and_resize(
        self, 
        images: np.ndarray, 
        bboxes: np.ndarray, 
        patch_size: int
    ) -> np.ndarray:
        """
        批量裁剪并调整大小
        
        根据边界框裁剪图像区域并调整到指定大小。
        等价于对每个边界框调用 get_foreground 函数。
        
        Args:
            images: 输入图像数组
                - 3D: [C, H, W] 单帧图像
                - 4D: [T, C, H, W] 多帧图像序列
            bboxes: 边界框数组 [N, 7] 格式为 [x1, y1, x2, y2, label, cx, cy]
            patch_size: 输出 patch 大小
        
        Returns:
            裁剪并调整大小后的图像 patches
            - 3D 输入: [N, C, patch_size, patch_size]
            - 4D 输入: [N, T, C, patch_size, patch_size]
        """
        if len(bboxes) == 0:
            if images.ndim == 3:
                return np.zeros((0, images.shape[0], patch_size, patch_size), dtype=images.dtype)
            else:
                return np.zeros((0, images.shape[0], images.shape[1], patch_size, patch_size), dtype=images.dtype)
        
        n_bboxes = len(bboxes)
        
        if images.ndim == 3:
            # 单帧图像 [C, H, W]
            c = images.shape[0]
            patches = np.zeros((n_bboxes, c, patch_size, patch_size), dtype=images.dtype)
            
            for i in range(n_bboxes):
                x_min = int(np.ceil(bboxes[i][0]))
                y_min = int(np.ceil(bboxes[i][1]))
                x_max = int(np.ceil(bboxes[i][2]))
                y_max = int(np.ceil(bboxes[i][3]))
                
                # 裁剪 [C, H, W] -> [C, h, w]
                crop = images[:, y_min:y_max, x_min:x_max]
                # 转换为 [h, w, C] 进行 resize
                crop_hwc = np.transpose(crop, (1, 2, 0))
                resized = cv2.resize(crop_hwc, (patch_size, patch_size), interpolation=self.interpolation)
                # 转回 [C, H, W]
                patches[i] = np.transpose(resized, (2, 0, 1))
            
            return patches
        
        elif images.ndim == 4:
            # 多帧图像序列 [T, C, H, W]
            t, c = images.shape[0], images.shape[1]
            patches = np.zeros((n_bboxes, t, c, patch_size, patch_size), dtype=images.dtype)
            
            for i in range(n_bboxes):
                x_min = int(np.ceil(bboxes[i][0]))
                y_min = int(np.ceil(bboxes[i][1]))
                x_max = int(np.ceil(bboxes[i][2]))
                y_max = int(np.ceil(bboxes[i][3]))
                
                for j in range(t):
                    # 裁剪 [C, H, W] -> [C, h, w]
                    crop = images[j, :, y_min:y_max, x_min:x_max]
                    # 转换为 [h, w, C] 进行 resize
                    crop_hwc = np.transpose(crop, (1, 2, 0))
                    resized = cv2.resize(crop_hwc, (patch_size, patch_size), interpolation=self.interpolation)
                    # 转回 [C, H, W]
                    patches[i, j] = np.transpose(resized, (2, 0, 1))
            
            return patches
        
        else:
            raise ValueError(f"Expected 3D or 4D array, got {images.ndim}D")


class MemoryMappedChunkedDataset(Dataset):
    """
    使用内存映射的分块数据集
    
    通过 numpy.memmap 实现内存映射加载，显著降低内存占用。
    支持预加载机制以提升数据访问效率。
    
    Attributes:
        chunk_files: List[str] - chunk 文件路径列表
        mmap_mode: str - 内存映射模式 ('r', 'r+', 'c')
        prefetch_chunks: int - 预加载的 chunk 数量
        last_flow: bool - 是否只使用最后一帧光流
        transform: callable - 数据变换函数
    
    Methods:
        __len__: 返回数据集大小
        __getitem__: 获取指定索引的数据
    """
    
    def __init__(
        self, 
        chunk_dir: str,
        mmap_mode: str = 'r',
        prefetch_chunks: int = 2,
        last_flow: bool = False,
        transform: Optional[callable] = transform
    ):
        """
        初始化内存映射分块数据集
        
        Args:
            chunk_dir: chunk 文件所在目录
            mmap_mode: 内存映射模式
                - 'r': 只读模式
                - 'r+': 读写模式
                - 'c': 写时复制模式
            prefetch_chunks: 预加载的 chunk 数量
            last_flow: 是否只使用最后一帧光流
            transform: 数据变换函数
        """
        super(MemoryMappedChunkedDataset, self).__init__()
        
        self.chunk_dir = chunk_dir
        self.mmap_mode = mmap_mode
        self.prefetch_chunks = prefetch_chunks
        self.last_flow = last_flow
        self.transform = transform
        
        # 查找所有 chunk 文件
        self.chunk_files = sorted(glob.glob(os.path.join(chunk_dir, "chunked_samples_*.pkl")))
        if not self.chunk_files:
            raise ValueError(f"No chunk files found in {chunk_dir}")
        
        # 创建临时目录存储 memmap 文件
        self.temp_dir = tempfile.mkdtemp(prefix="hf2vad_mmap_")
        
        # 加载 chunk 元数据并创建 memmap 文件
        self._chunk_metadata: List[Dict[str, Any]] = []
        self._chunk_lengths: List[int] = []
        self._cumulative_lengths: List[int] = []
        self._mmap_files: Dict[int, Dict[str, np.memmap]] = {}
        
        self._initialize_chunks()
        
        # 预加载队列
        self._prefetch_queue: Queue = Queue(maxsize=prefetch_chunks)
        self._current_chunk_idx: int = -1
        
        # 预加载第一个 chunk
        if len(self.chunk_files) > 0:
            self._load_chunk(0)
    
    def _initialize_chunks(self):
        """初始化所有 chunk 的元数据"""
        cumulative = 0
        
        for chunk_idx, chunk_file in enumerate(self.chunk_files):
            # 加载 chunk 获取元数据
            chunk_data = joblib.load(chunk_file)
            chunk_len = len(chunk_data["sample_id"])
            
            self._chunk_metadata.append({
                'file': chunk_file,
                'length': chunk_len,
                'shapes': {
                    'appearance': chunk_data["appearance"].shape,
                    'motion': chunk_data["motion"].shape,
                    'bbox': chunk_data["bbox"].shape,
                    'pred_frame': chunk_data["pred_frame"].shape,
                    'sample_id': chunk_data["sample_id"].shape,
                }
            })
            
            self._chunk_lengths.append(chunk_len)
            cumulative += chunk_len
            self._cumulative_lengths.append(cumulative)
            
            # 创建 memmap 文件
            self._create_memmap_for_chunk(chunk_idx, chunk_data)
            
            # 释放原始数据
            del chunk_data
    
    def _create_memmap_for_chunk(self, chunk_idx: int, chunk_data: Dict[str, np.ndarray]):
        """为指定 chunk 创建 memmap 文件"""
        mmap_files = {}
        
        for key in ['appearance', 'motion', 'bbox', 'pred_frame', 'sample_id']:
            data = chunk_data[key]
            mmap_path = os.path.join(self.temp_dir, f"chunk_{chunk_idx}_{key}.npy")
            
            # 保存为 npy 文件
            np.save(mmap_path, data)
            
            # 创建 memmap
            mmap_files[key] = np.load(mmap_path, mmap_mode=self.mmap_mode)
        
        self._mmap_files[chunk_idx] = mmap_files
    
    def _load_chunk(self, chunk_idx: int):
        """加载指定的 chunk（如果尚未加载）"""
        if chunk_idx not in self._mmap_files:
            chunk_data = joblib.load(self.chunk_files[chunk_idx])
            self._create_memmap_for_chunk(chunk_idx, chunk_data)
            del chunk_data
        
        self._current_chunk_idx = chunk_idx
        
        # 预加载后续 chunks
        self._prefetch_next_chunks(chunk_idx)
    
    def _prefetch_next_chunks(self, current_idx: int):
        """预加载后续的 chunks"""
        for i in range(1, self.prefetch_chunks + 1):
            next_idx = current_idx + i
            if next_idx < len(self.chunk_files) and next_idx not in self._mmap_files:
                # 在后台线程中预加载
                def prefetch(idx):
                    chunk_data = joblib.load(self.chunk_files[idx])
                    self._create_memmap_for_chunk(idx, chunk_data)
                    del chunk_data
                
                thread = Thread(target=prefetch, args=(next_idx,))
                thread.daemon = True
                thread.start()
    
    def _get_chunk_and_local_idx(self, global_idx: int) -> Tuple[int, int]:
        """将全局索引转换为 chunk 索引和局部索引"""
        for chunk_idx, cumulative in enumerate(self._cumulative_lengths):
            if global_idx < cumulative:
                local_idx = global_idx
                if chunk_idx > 0:
                    local_idx = global_idx - self._cumulative_lengths[chunk_idx - 1]
                return chunk_idx, local_idx
        
        raise IndexError(f"Index {global_idx} out of range")
    
    def __len__(self) -> int:
        """返回数据集总大小"""
        return self._cumulative_lengths[-1] if self._cumulative_lengths else 0
    
    def __getitem__(self, indice: int) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, int]:
        """
        获取指定索引的数据
        
        Args:
            indice: 数据索引
        
        Returns:
            Tuple containing:
                - appearance: 外观特征张量
                - motion: 运动特征张量
                - bbox: 边界框数据
                - pred_frame: 预测帧索引
                - indice: 原始索引
        """
        chunk_idx, local_idx = self._get_chunk_and_local_idx(indice)
        
        # 确保 chunk 已加载
        if chunk_idx not in self._mmap_files:
            self._load_chunk(chunk_idx)
        
        mmap_data = self._mmap_files[chunk_idx]
        
        # 读取数据（通过 memmap 按需加载）
        appearance = np.array(mmap_data['appearance'][local_idx])
        motion = np.array(mmap_data['motion'][local_idx])
        bbox = np.array(mmap_data['bbox'][local_idx])
        pred_frame = np.array(mmap_data['pred_frame'][local_idx])
        
        # 处理 appearance: [#frame, h, w, c] -> [h, w, #frame*c]
        x = np.transpose(appearance, [1, 2, 0, 3])
        x = np.reshape(x, (x.shape[0], x.shape[1], -1))
        
        # 处理 motion
        y = motion[1:] if not self.last_flow else motion[-1:]
        y = np.transpose(y, [1, 2, 0, 3])
        y = np.reshape(y, (y.shape[0], y.shape[1], -1))
        
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        
        return x, y, bbox.astype(np.float32), pred_frame, indice
    
    def __del__(self):
        """清理临时文件"""
        try:
            # 关闭所有 memmap
            for chunk_mmap in self._mmap_files.values():
                for mmap in chunk_mmap.values():
                    del mmap
            self._mmap_files.clear()
            
            # 删除临时目录
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass


class SingleChunkMemoryMappedDataset(Dataset):
    """
    单个 chunk 文件的内存映射数据集
    
    用于兼容原始的 Chunked_sample_dataset 接口，
    但使用内存映射方式加载数据。
    
    Attributes:
        chunk_file: str - chunk 文件路径
        last_flow: bool - 是否只使用最后一帧光流
        transform: callable - 数据变换函数
    """
    
    def __init__(
        self, 
        chunk_file: str, 
        last_flow: bool = False, 
        transform: Optional[callable] = transform,
        mmap_mode: str = 'r'
    ):
        """
        初始化单 chunk 内存映射数据集
        
        Args:
            chunk_file: chunk 文件路径
            last_flow: 是否只使用最后一帧光流
            transform: 数据变换函数
            mmap_mode: 内存映射模式
        """
        super(SingleChunkMemoryMappedDataset, self).__init__()
        
        self.chunk_file = chunk_file
        self.last_flow = last_flow
        self.transform = transform
        self.mmap_mode = mmap_mode
        
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp(prefix="hf2vad_single_mmap_")
        
        # 加载并创建 memmap
        self._load_and_create_memmap()
    
    def _load_and_create_memmap(self):
        """加载 chunk 并创建 memmap 文件"""
        chunk_data = joblib.load(self.chunk_file)
        
        self._mmap_files = {}
        self._length = len(chunk_data["sample_id"])
        
        for key in ['appearance', 'motion', 'bbox', 'pred_frame', 'sample_id']:
            data = chunk_data[key]
            mmap_path = os.path.join(self.temp_dir, f"{key}.npy")
            
            # 保存为 npy 文件
            np.save(mmap_path, data)
            
            # 创建 memmap
            self._mmap_files[key] = np.load(mmap_path, mmap_mode=self.mmap_mode)
        
        del chunk_data
    
    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, indice: int) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, int]:
        """
        获取指定索引的数据
        
        与原始 Chunked_sample_dataset 接口完全兼容
        """
        appearance = np.array(self._mmap_files['appearance'][indice])
        motion = np.array(self._mmap_files['motion'][indice])
        bbox = np.array(self._mmap_files['bbox'][indice])
        pred_frame = np.array(self._mmap_files['pred_frame'][indice])
        
        # [#frame, h, w, c] -> [h, w, #frame*c]
        x = np.transpose(appearance, [1, 2, 0, 3])
        x = np.reshape(x, (x.shape[0], x.shape[1], -1))
        
        y = motion[1:] if not self.last_flow else motion[-1:]
        y = np.transpose(y, [1, 2, 0, 3])
        y = np.reshape(y, (y.shape[0], y.shape[1], -1))
        
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        
        return x, y, bbox.astype(np.float32), pred_frame, indice
    
    def __del__(self):
        """清理临时文件"""
        try:
            for mmap in self._mmap_files.values():
                del mmap
            self._mmap_files.clear()
            
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass




class ChunkedIterableDataset(IterableDataset):
    """
    支持跨 chunk 无缝数据流的可迭代数据集
    
    实现了跨 chunk 文件边界的无缝数据流，
    支持多 chunk 预加载以减少 I/O 等待时间。
    支持多 worker 并行加载，每个 worker 处理不同的 chunk 子集。
    
    Attributes:
        chunk_dir: str - chunk 文件目录
        prefetch_chunks: int - 预加载的 chunk 数量
        last_flow: bool - 是否只使用最后一帧光流
        shuffle: bool - 是否打乱数据
        shuffle_chunks: bool - 是否打乱 chunk 顺序
        transform: callable - 数据变换函数
        seed: int - 随机种子，用于可复现性
    
    Requirements:
        - 2.1: 预加载多个 chunk 文件
        - 2.3: 支持跨 chunk 文件的无缝数据流
    """
    
    def __init__(
        self,
        chunk_dir: str,
        prefetch_chunks: int = 2,
        last_flow: bool = False,
        shuffle: bool = True,
        shuffle_chunks: bool = True,
        transform: Optional[callable] = transform,
        seed: Optional[int] = None
    ):
        """
        初始化可迭代数据集
        
        Args:
            chunk_dir: chunk 文件目录
            prefetch_chunks: 预加载的 chunk 数量
            last_flow: 是否只使用最后一帧光流
            shuffle: 是否打乱 chunk 内的数据
            shuffle_chunks: 是否打乱 chunk 顺序
            transform: 数据变换函数
            seed: 随机种子
        """
        super(ChunkedIterableDataset, self).__init__()
        
        self.chunk_dir = chunk_dir
        self.prefetch_chunks = prefetch_chunks
        self.last_flow = last_flow
        self.shuffle = shuffle
        self.shuffle_chunks = shuffle_chunks
        self.transform = transform
        self.seed = seed
        
        # 查找所有 chunk 文件
        self.chunk_files = sorted(glob.glob(os.path.join(chunk_dir, "chunked_samples_*.pkl")))
        if not self.chunk_files:
            raise ValueError(f"No chunk files found in {chunk_dir}")
        
        # 计算每个 chunk 的长度和总长度
        self._chunk_lengths: List[int] = []
        self._total_length = 0
        for chunk_file in self.chunk_files:
            chunk_data = joblib.load(chunk_file)
            chunk_len = len(chunk_data["sample_id"])
            self._chunk_lengths.append(chunk_len)
            self._total_length += chunk_len
            del chunk_data
        
        # 计算累积长度，用于全局索引计算
        self._cumulative_lengths: List[int] = []
        cumulative = 0
        for length in self._chunk_lengths:
            cumulative += length
            self._cumulative_lengths.append(cumulative)
    
    def __len__(self) -> int:
        return self._total_length
    
    def _get_worker_info(self) -> Tuple[int, int]:
        """
        获取当前 worker 的信息
        
        Returns:
            Tuple of (worker_id, num_workers)
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # 单进程模式
            return 0, 1
        return worker_info.id, worker_info.num_workers
    
    def _get_chunk_indices_for_worker(self, worker_id: int, num_workers: int) -> List[int]:
        """
        获取当前 worker 应该处理的 chunk 索引列表
        
        将 chunk 文件均匀分配给各个 worker，确保无数据重复或丢失。
        
        Args:
            worker_id: 当前 worker ID
            num_workers: 总 worker 数量
        
        Returns:
            当前 worker 应处理的 chunk 索引列表
        """
        all_chunk_indices = list(range(len(self.chunk_files)))
        
        # 打乱 chunk 顺序（如果启用）
        if self.shuffle_chunks:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(all_chunk_indices)
        
        # 将 chunks 分配给各个 worker
        # 使用轮询方式分配，确保负载均衡
        worker_chunk_indices = []
        for i, chunk_idx in enumerate(all_chunk_indices):
            if i % num_workers == worker_id:
                worker_chunk_indices.append(chunk_idx)
        
        return worker_chunk_indices
    
    def _prefetch_worker(
        self, 
        chunk_files: List[str], 
        queue: Queue,
        stop_event: Any
    ):
        """
        预加载 worker 线程函数
        
        Args:
            chunk_files: 要加载的 chunk 文件列表
            queue: 预加载队列
            stop_event: 停止事件
        """
        for chunk_file in chunk_files:
            if stop_event.is_set():
                break
            try:
                chunk_data = joblib.load(chunk_file)
                queue.put(chunk_data)
            except Exception as e:
                # 记录错误但继续处理
                print(f"Warning: Failed to load chunk {chunk_file}: {e}")
                continue
        queue.put(None)  # 结束信号
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, int]]:
        """
        迭代数据集
        
        支持多 worker 并行加载，每个 worker 处理不同的 chunk 子集。
        实现跨 chunk 文件的无缝数据流。
        
        Yields:
            Tuple containing appearance, motion, bbox, pred_frame, global_idx
        """
        import threading
        
        worker_id, num_workers = self._get_worker_info()
        
        # 获取当前 worker 应处理的 chunk 索引
        chunk_indices = self._get_chunk_indices_for_worker(worker_id, num_workers)
        
        if not chunk_indices:
            return  # 当前 worker 没有分配到任何 chunk
        
        # 获取对应的 chunk 文件
        chunk_files_to_load = [self.chunk_files[i] for i in chunk_indices]
        
        # 预加载队列
        prefetch_queue: Queue = Queue(maxsize=self.prefetch_chunks)
        stop_event = threading.Event()
        
        # 启动预加载线程
        prefetch_thread = Thread(
            target=self._prefetch_worker, 
            args=(chunk_files_to_load, prefetch_queue, stop_event)
        )
        prefetch_thread.daemon = True
        prefetch_thread.start()
        
        # 计算全局索引偏移
        # 对于多 worker 模式，需要正确计算全局索引
        global_idx_offset = 0
        for i in range(chunk_indices[0]):
            global_idx_offset += self._chunk_lengths[i]
        
        current_chunk_list_idx = 0
        
        try:
            while True:
                chunk_data = prefetch_queue.get()
                if chunk_data is None:
                    break
                
                # 获取当前 chunk 的数据
                appearances = chunk_data["appearance"]
                motions = chunk_data["motion"]
                bboxes = chunk_data["bbox"]
                pred_frames = chunk_data["pred_frame"]
                sample_ids = chunk_data["sample_id"]
                
                chunk_len = len(sample_ids)
                
                # 计算当前 chunk 的全局索引起始位置
                if current_chunk_list_idx < len(chunk_indices):
                    actual_chunk_idx = chunk_indices[current_chunk_list_idx]
                    chunk_global_start = 0
                    for i in range(actual_chunk_idx):
                        chunk_global_start += self._chunk_lengths[i]
                else:
                    chunk_global_start = global_idx_offset
                
                # 生成索引
                indices = np.arange(chunk_len)
                if self.shuffle:
                    # 使用基于 worker_id 和 chunk_idx 的种子确保可复现性
                    local_seed = (self.seed or 0) + worker_id * 1000 + current_chunk_list_idx
                    rng = np.random.RandomState(local_seed)
                    rng.shuffle(indices)
                
                for local_idx in indices:
                    appearance = appearances[local_idx]
                    motion = motions[local_idx]
                    bbox = bboxes[local_idx]
                    pred_frame = pred_frames[local_idx]
                    
                    # 处理数据
                    x = np.transpose(appearance, [1, 2, 0, 3])
                    x = np.reshape(x, (x.shape[0], x.shape[1], -1))
                    
                    y = motion[1:] if not self.last_flow else motion[-1:]
                    y = np.transpose(y, [1, 2, 0, 3])
                    y = np.reshape(y, (y.shape[0], y.shape[1], -1))
                    
                    if self.transform:
                        x = self.transform(x)
                        y = self.transform(y)
                    
                    # 计算全局索引
                    global_idx = chunk_global_start + local_idx
                    
                    yield x, y, bbox.astype(np.float32), pred_frame, global_idx
                
                del chunk_data
                current_chunk_list_idx += 1
                
        finally:
            # 清理
            stop_event.set()
            # 清空队列
            while not prefetch_queue.empty():
                try:
                    prefetch_queue.get_nowait()
                except:
                    break


class AutoNumWorkersConfig:
    """
    自动 num_workers 配置类
    
    根据系统资源（CPU 核心数、可用内存）自动计算最优的 num_workers 参数。
    
    Attributes:
        cpu_count: int - CPU 核心数
        available_memory_gb: float - 可用内存（GB）
        estimated_sample_memory_mb: float - 估计每个样本的内存占用（MB）
        max_workers: int - 最大 worker 数量限制
        min_workers: int - 最小 worker 数量
    
    Requirements:
        - 2.2: 根据 CPU 核心数和内存自动配置 num_workers
    """
    
    def __init__(
        self,
        estimated_sample_memory_mb: float = 50.0,
        max_workers: int = 16,
        min_workers: int = 0,
        memory_safety_factor: float = 0.7,
        cpu_usage_factor: float = 0.75
    ):
        """
        初始化自动配置
        
        Args:
            estimated_sample_memory_mb: 估计每个样本的内存占用（MB）
            max_workers: 最大 worker 数量限制
            min_workers: 最小 worker 数量
            memory_safety_factor: 内存安全系数（0-1），避免使用全部可用内存
            cpu_usage_factor: CPU 使用系数（0-1），避免使用全部 CPU 核心
        """
        self.estimated_sample_memory_mb = estimated_sample_memory_mb
        self.max_workers = max_workers
        self.min_workers = min_workers
        self.memory_safety_factor = memory_safety_factor
        self.cpu_usage_factor = cpu_usage_factor
        
        # 获取系统信息
        self.cpu_count = self._get_cpu_count()
        self.available_memory_gb = self._get_available_memory_gb()
    
    def _get_cpu_count(self) -> int:
        """获取 CPU 核心数"""
        try:
            import multiprocessing
            return multiprocessing.cpu_count()
        except Exception:
            return 4  # 默认值
    
    def _get_available_memory_gb(self) -> float:
        """获取可用内存（GB）"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return mem.available / (1024 ** 3)
        except ImportError:
            # psutil 未安装，尝试其他方法
            try:
                import os
                if os.name == 'nt':  # Windows
                    import ctypes
                    
                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ('dwLength', ctypes.c_ulong),
                            ('dwMemoryLoad', ctypes.c_ulong),
                            ('ullTotalPhys', ctypes.c_ulonglong),
                            ('ullAvailPhys', ctypes.c_ulonglong),
                            ('ullTotalPageFile', ctypes.c_ulonglong),
                            ('ullAvailPageFile', ctypes.c_ulonglong),
                            ('ullTotalVirtual', ctypes.c_ulonglong),
                            ('ullAvailVirtual', ctypes.c_ulonglong),
                            ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                        ]
                    
                    memoryStatus = MEMORYSTATUSEX()
                    memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memoryStatus))
                    return memoryStatus.ullAvailPhys / (1024 ** 3)
                else:  # Linux/Unix
                    with open('/proc/meminfo', 'r') as f:
                        for line in f:
                            if 'MemAvailable' in line:
                                return int(line.split()[1]) / (1024 ** 2)
                            elif 'MemFree' in line:
                                # 如果没有 MemAvailable，使用 MemFree
                                return int(line.split()[1]) / (1024 ** 2)
            except Exception:
                pass
            return 8.0  # 默认 8GB
    
    def calculate_optimal_workers(
        self,
        batch_size: int = 128,
        prefetch_factor: int = 2
    ) -> int:
        """
        计算最优的 num_workers 数量
        
        考虑因素：
        1. CPU 核心数：不超过 CPU 核心数的一定比例
        2. 可用内存：确保有足够内存供 worker 使用
        3. 批次大小：较大的批次可能需要更多 worker
        
        Args:
            batch_size: 批次大小
            prefetch_factor: 预取因子（每个 worker 预取的批次数）
        
        Returns:
            推荐的 num_workers 数量
        """
        # 基于 CPU 核心数的限制
        cpu_based_workers = int(self.cpu_count * self.cpu_usage_factor)
        
        # 基于内存的限制
        # 每个 worker 需要的内存 = batch_size * prefetch_factor * sample_memory
        memory_per_worker_gb = (
            batch_size * prefetch_factor * self.estimated_sample_memory_mb
        ) / 1024
        
        safe_available_memory = self.available_memory_gb * self.memory_safety_factor
        memory_based_workers = int(safe_available_memory / max(memory_per_worker_gb, 0.1))
        
        # 取两者的最小值
        optimal_workers = min(cpu_based_workers, memory_based_workers)
        
        # 应用最大最小限制
        optimal_workers = max(self.min_workers, min(optimal_workers, self.max_workers))
        
        return optimal_workers
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        获取系统信息摘要
        
        Returns:
            包含系统信息的字典
        """
        return {
            'cpu_count': self.cpu_count,
            'available_memory_gb': round(self.available_memory_gb, 2),
            'estimated_sample_memory_mb': self.estimated_sample_memory_mb,
            'max_workers': self.max_workers,
            'min_workers': self.min_workers,
        }


def get_auto_num_workers(
    batch_size: int = 128,
    estimated_sample_memory_mb: float = 50.0,
    max_workers: int = 16,
    min_workers: int = 0,
    prefetch_factor: int = 2,
    verbose: bool = False
) -> int:
    """
    自动计算最优的 num_workers 数量
    
    根据系统的 CPU 核心数和可用内存自动计算最优的 DataLoader num_workers 参数。
    
    Args:
        batch_size: 批次大小
        estimated_sample_memory_mb: 估计每个样本的内存占用（MB）
        max_workers: 最大 worker 数量限制
        min_workers: 最小 worker 数量
        prefetch_factor: 预取因子
        verbose: 是否打印详细信息
    
    Returns:
        推荐的 num_workers 数量
    
    Requirements:
        - 2.2: 根据 CPU 核心数和内存自动配置
    
    Example:
        >>> num_workers = get_auto_num_workers(batch_size=128)
        >>> dataloader = DataLoader(dataset, batch_size=128, num_workers=num_workers)
    """
    config = AutoNumWorkersConfig(
        estimated_sample_memory_mb=estimated_sample_memory_mb,
        max_workers=max_workers,
        min_workers=min_workers
    )
    
    optimal_workers = config.calculate_optimal_workers(
        batch_size=batch_size,
        prefetch_factor=prefetch_factor
    )
    
    if verbose:
        info = config.get_system_info()
        print(f"System Info:")
        print(f"  CPU cores: {info['cpu_count']}")
        print(f"  Available memory: {info['available_memory_gb']:.2f} GB")
        print(f"  Estimated sample memory: {info['estimated_sample_memory_mb']:.1f} MB")
        print(f"  Recommended num_workers: {optimal_workers}")
    
    return optimal_workers


def get_optimized_dataloader(
    chunk_dir: str,
    batch_size: int = 128,
    num_workers: Optional[int] = None,
    shuffle: bool = True,
    last_flow: bool = False,
    use_iterable: bool = False,
    prefetch_chunks: int = 2,
    mmap_mode: str = 'r',
    auto_num_workers: bool = True,
    seed: Optional[int] = None
) -> DataLoader:
    """
    获取优化的数据加载器
    
    根据参数选择使用 MemoryMappedChunkedDataset 或 ChunkedIterableDataset。
    支持自动配置 num_workers。
    
    Args:
        chunk_dir: chunk 文件目录
        batch_size: 批次大小
        num_workers: 工作进程数，如果为 None 且 auto_num_workers=True，则自动计算
        shuffle: 是否打乱数据
        last_flow: 是否只使用最后一帧光流
        use_iterable: 是否使用可迭代数据集（支持跨 chunk 无缝数据流）
        prefetch_chunks: 预加载的 chunk 数量
        mmap_mode: 内存映射模式
        auto_num_workers: 是否自动计算 num_workers
        seed: 随机种子，用于可复现性
    
    Returns:
        配置好的 DataLoader
    
    Requirements:
        - 2.1: 预加载多个 chunk 文件
        - 2.2: 根据系统资源自动调整 num_workers
        - 2.3: 支持跨 chunk 文件的无缝数据流
    """
    # 自动计算 num_workers
    if num_workers is None:
        if auto_num_workers:
            num_workers = get_auto_num_workers(
                batch_size=batch_size,
                verbose=False
            )
        else:
            num_workers = 4  # 默认值
    
    if use_iterable:
        dataset = ChunkedIterableDataset(
            chunk_dir=chunk_dir,
            prefetch_chunks=prefetch_chunks,
            last_flow=last_flow,
            shuffle=shuffle,
            shuffle_chunks=shuffle,
            transform=transform,
            seed=seed
        )
        # IterableDataset 不支持 shuffle 参数
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
    else:
        dataset = MemoryMappedChunkedDataset(
            chunk_dir=chunk_dir,
            mmap_mode=mmap_mode,
            prefetch_chunks=prefetch_chunks,
            last_flow=last_flow,
            transform=transform
        )
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle
        )


# 导出的类和函数
__all__ = [
    'BatchImageProcessor',
    'MemoryMappedChunkedDataset',
    'SingleChunkMemoryMappedDataset',
    'ChunkedIterableDataset',
    'get_optimized_dataloader',
    'AutoNumWorkersConfig',
    'get_auto_num_workers',
]
