"""
多精度训练工具模块

提供 FP32 和 FP64 精度训练支持，包括：
- PrecisionTrainer: 多精度训练器
- FP64ModelWrapper: FP64 模型包装器

Requirements: 4.1, 4.2, 4.5
"""

import warnings
from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


# 支持的精度类型
SUPPORTED_PRECISIONS = ("fp32", "fp64")


@dataclass
class PrecisionConfig:
    """
    精度配置数据类
    
    Attributes:
        precision: 精度类型 ('fp32' 或 'fp64')
        auto_fallback: 是否在不支持时自动回退
    """
    precision: str = "fp32"
    auto_fallback: bool = True
    
    def __post_init__(self):
        if self.precision not in SUPPORTED_PRECISIONS:
            raise ValueError(
                f"不支持的精度类型: {self.precision}. "
                f"支持的类型: {SUPPORTED_PRECISIONS}"
            )
    
    @property
    def dtype(self) -> torch.dtype:
        """获取对应的 torch dtype"""
        return torch.float64 if self.precision == "fp64" else torch.float32


def check_fp64_support(device: Union[str, torch.device]) -> bool:
    """
    检查设备是否支持 FP64 计算
    
    Args:
        device: 目标设备
        
    Returns:
        是否支持 FP64
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    if device.type == "cpu":
        return True
    
    if device.type == "cuda":
        if not torch.cuda.is_available():
            return False
        
        device_idx = device.index if device.index is not None else 0
        capability = torch.cuda.get_device_capability(device_idx)
        # 大多数现代 GPU 都支持 FP64，但性能可能较差
        # 这里我们假设所有 CUDA 设备都支持 FP64
        return True
    
    return False


class FP64ModelWrapper(nn.Module):
    """
    FP64 模型包装器，自动将模型转换为双精度
    
    该包装器会将模型的所有参数和缓冲区转换为 FP64，
    并确保前向传播时输入数据也被转换为 FP64。
    
    Attributes:
        model: 原始模型
        _is_fp64: 当前是否为 FP64 模式
        
    Example:
        >>> model = MyModel()
        >>> fp64_model = FP64ModelWrapper(model)
        >>> fp64_model.convert_to_fp64()
        >>> output = fp64_model(input_tensor)  # 自动转换输入为 FP64
    """
    
    def __init__(self, model: nn.Module):
        """
        初始化 FP64 模型包装器
        
        Args:
            model: 要包装的原始模型
        """
        super(FP64ModelWrapper, self).__init__()
        self.model = model
        self._is_fp64 = False
    
    def convert_to_fp64(self) -> "FP64ModelWrapper":
        """
        将模型转换为 FP64 精度
        
        Returns:
            self，支持链式调用
        """
        self.model = self.model.double()
        self._is_fp64 = True
        return self
    
    def convert_to_fp32(self) -> "FP64ModelWrapper":
        """
        将模型转换为 FP32 精度
        
        Returns:
            self，支持链式调用
        """
        self.model = self.model.float()
        self._is_fp64 = False
        return self
    
    @property
    def is_fp64(self) -> bool:
        """检查当前是否为 FP64 模式"""
        return self._is_fp64
    
    @property
    def dtype(self) -> torch.dtype:
        """获取当前精度类型"""
        return torch.float64 if self._is_fp64 else torch.float32
    
    def _convert_input(self, x: Any) -> Any:
        """
        递归转换输入数据的精度
        
        Args:
            x: 输入数据（可以是 Tensor、dict、list、tuple）
            
        Returns:
            转换后的数据
        """
        if isinstance(x, torch.Tensor):
            if x.is_floating_point():
                return x.to(self.dtype)
            return x
        elif isinstance(x, dict):
            return {k: self._convert_input(v) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            converted = [self._convert_input(item) for item in x]
            return type(x)(converted)
        return x
    
    def forward(self, *args, **kwargs) -> Any:
        """
        前向传播，自动转换输入精度
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            模型输出
        """
        # 转换所有输入参数的精度
        converted_args = tuple(self._convert_input(arg) for arg in args)
        converted_kwargs = {k: self._convert_input(v) for k, v in kwargs.items()}
        
        return self.model(*converted_args, **converted_kwargs)
    
    def __getattr__(self, name: str) -> Any:
        """
        代理属性访问到内部模型
        
        Args:
            name: 属性名
            
        Returns:
            属性值
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
    
    def state_dict(self, *args, **kwargs):
        """获取模型状态字典"""
        return self.model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        """加载模型状态字典"""
        return self.model.load_state_dict(state_dict, *args, **kwargs)
    
    def parameters(self, *args, **kwargs):
        """获取模型参数"""
        return self.model.parameters(*args, **kwargs)
    
    def named_parameters(self, *args, **kwargs):
        """获取命名参数"""
        return self.model.named_parameters(*args, **kwargs)
    
    def train(self, mode: bool = True):
        """设置训练模式"""
        self.model.train(mode)
        return self
    
    def eval(self):
        """设置评估模式"""
        self.model.eval()
        return self


class PrecisionTrainer:
    """
    多精度训练器，支持 FP32 和 FP64 训练
    
    该训练器提供统一的接口来处理不同精度的训练，
    包括自动精度转换、检查点保存/加载等功能。
    
    Attributes:
        model: 训练模型
        optimizer: 优化器
        precision: 精度类型 ('fp32' 或 'fp64')
        device: 训练设备
        
    Example:
        >>> model = MyModel()
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> trainer = PrecisionTrainer(model, optimizer, precision='fp64')
        >>> loss_dict = trainer.train_step(batch)
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        precision: str = "fp32",
        device: str = "cuda:0",
        auto_fallback: bool = True
    ):
        """
        初始化多精度训练器
        
        Args:
            model: 训练模型
            optimizer: 优化器
            precision: 精度类型 ('fp32' 或 'fp64')
            device: 训练设备
            auto_fallback: 如果设备不支持 FP64，是否自动回退到 FP32
            
        Raises:
            ValueError: 不支持的精度类型
            RuntimeError: 设备不支持 FP64 且未启用自动回退
        """
        if precision not in SUPPORTED_PRECISIONS:
            raise ValueError(
                f"不支持的精度类型: {precision}. "
                f"支持的类型: {SUPPORTED_PRECISIONS}"
            )
        
        self.device = torch.device(device)
        self._original_precision = precision
        self.precision = precision
        self.auto_fallback = auto_fallback
        
        # 检查 FP64 支持
        if precision == "fp64":
            if not check_fp64_support(self.device):
                if auto_fallback:
                    warnings.warn(
                        f"设备 {device} 不支持 FP64 计算，自动回退到 FP32",
                        RuntimeWarning
                    )
                    self.precision = "fp32"
                else:
                    raise RuntimeError(
                        f"设备 {device} 不支持 FP64 计算"
                    )
        
        # 包装模型
        self._wrapped_model = FP64ModelWrapper(model)
        
        # 设置精度
        if self.precision == "fp64":
            self._wrapped_model.convert_to_fp64()
        else:
            self._wrapped_model.convert_to_fp32()
        
        # 移动到设备
        self._wrapped_model = self._wrapped_model.to(self.device)
        
        # 保存优化器引用
        self.optimizer = optimizer
        
        # 更新优化器参数组的精度
        self._update_optimizer_precision()
    
    def _update_optimizer_precision(self):
        """更新优化器参数组以匹配当前精度"""
        # 优化器的参数引用会自动更新，因为它们指向同一个参数对象
        pass
    
    @property
    def model(self) -> nn.Module:
        """获取内部模型（不包含包装器）"""
        return self._wrapped_model.model
    
    @property
    def wrapped_model(self) -> FP64ModelWrapper:
        """获取包装后的模型"""
        return self._wrapped_model
    
    @property
    def dtype(self) -> torch.dtype:
        """获取当前精度对应的 dtype"""
        return torch.float64 if self.precision == "fp64" else torch.float32
    
    @property
    def is_fp64(self) -> bool:
        """检查是否为 FP64 模式"""
        return self.precision == "fp64"
    
    def to_precision(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        将张量转换为当前精度
        
        Args:
            tensor: 输入张量
            
        Returns:
            转换后的张量
        """
        if tensor.is_floating_point():
            return tensor.to(dtype=self.dtype, device=self.device)
        return tensor.to(device=self.device)
    
    def to_device(self, data: Any) -> Any:
        """
        将数据移动到设备并转换精度
        
        Args:
            data: 输入数据（可以是 Tensor、dict、list、tuple）
            
        Returns:
            转换后的数据
        """
        if isinstance(data, torch.Tensor):
            return self.to_precision(data)
        elif isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            converted = [self.to_device(item) for item in data]
            return type(data)(converted)
        return data
    
    def train_step(
        self,
        batch: Tuple,
        loss_fn: Optional[callable] = None,
        **loss_kwargs
    ) -> Dict[str, float]:
        """
        执行单步训练
        
        Args:
            batch: 训练批次数据
            loss_fn: 损失函数（可选，如果模型自带损失计算）
            **loss_kwargs: 传递给损失函数的额外参数
            
        Returns:
            包含损失值的字典
        """
        self._wrapped_model.train()
        
        # 转换批次数据精度
        batch = self.to_device(batch)
        
        # 前向传播
        if isinstance(batch, (list, tuple)):
            outputs = self._wrapped_model(*batch)
        elif isinstance(batch, dict):
            outputs = self._wrapped_model(**batch)
        else:
            outputs = self._wrapped_model(batch)
        
        # 计算损失
        if loss_fn is not None:
            if isinstance(outputs, dict):
                loss = loss_fn(outputs, **loss_kwargs)
            else:
                loss = loss_fn(outputs, **loss_kwargs)
        elif isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        else:
            raise ValueError(
                "需要提供 loss_fn 或模型输出中包含 'loss' 键"
            )
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 返回损失值
        if isinstance(loss, torch.Tensor):
            return {"loss": loss.item()}
        return {"loss": float(loss)}
    
    def validate(
        self,
        dataloader: DataLoader,
        loss_fn: Optional[callable] = None,
        **loss_kwargs
    ) -> Dict[str, float]:
        """
        在验证集上评估模型
        
        Args:
            dataloader: 验证数据加载器
            loss_fn: 损失函数
            **loss_kwargs: 传递给损失函数的额外参数
            
        Returns:
            包含平均损失值的字典
        """
        self._wrapped_model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # 转换批次数据精度
                batch = self.to_device(batch)
                
                # 前向传播
                if isinstance(batch, (list, tuple)):
                    outputs = self._wrapped_model(*batch)
                elif isinstance(batch, dict):
                    outputs = self._wrapped_model(**batch)
                else:
                    outputs = self._wrapped_model(batch)
                
                # 计算损失
                if loss_fn is not None:
                    if isinstance(outputs, dict):
                        loss = loss_fn(outputs, **loss_kwargs)
                    else:
                        loss = loss_fn(outputs, **loss_kwargs)
                elif isinstance(outputs, dict) and "loss" in outputs:
                    loss = outputs["loss"]
                else:
                    continue
                
                if isinstance(loss, torch.Tensor):
                    total_loss += loss.item()
                else:
                    total_loss += float(loss)
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return {"val_loss": avg_loss}
    
    def save_checkpoint(
        self,
        path: str,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        **extra_state
    ) -> None:
        """
        保存训练检查点
        
        Args:
            path: 保存路径
            epoch: 当前轮数
            step: 当前步数
            **extra_state: 额外的状态信息
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "precision": self.precision,
            "epoch": epoch,
            "step": step,
        }
        checkpoint.update(extra_state)
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(
        self,
        path: str,
        load_optimizer: bool = True
    ) -> Dict[str, Any]:
        """
        加载训练检查点
        
        Args:
            path: 检查点路径
            load_optimizer: 是否加载优化器状态
            
        Returns:
            检查点中的额外状态信息
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # 加载模型状态
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # 加载优化器状态
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # 检查精度是否匹配
        saved_precision = checkpoint.get("precision", "fp32")
        if saved_precision != self.precision:
            warnings.warn(
                f"检查点精度 ({saved_precision}) 与当前精度 ({self.precision}) 不匹配，"
                f"模型已转换为当前精度"
            )
        
        # 返回额外状态
        extra_state = {
            k: v for k, v in checkpoint.items()
            if k not in ["model_state_dict", "optimizer_state_dict", "precision"]
        }
        return extra_state
    
    def set_precision(self, precision: str) -> None:
        """
        动态设置精度
        
        Args:
            precision: 新的精度类型 ('fp32' 或 'fp64')
        """
        if precision not in SUPPORTED_PRECISIONS:
            raise ValueError(
                f"不支持的精度类型: {precision}. "
                f"支持的类型: {SUPPORTED_PRECISIONS}"
            )
        
        if precision == self.precision:
            return
        
        if precision == "fp64":
            if not check_fp64_support(self.device):
                if self.auto_fallback:
                    warnings.warn(
                        f"设备 {self.device} 不支持 FP64 计算，保持 FP32",
                        RuntimeWarning
                    )
                    return
                else:
                    raise RuntimeError(
                        f"设备 {self.device} 不支持 FP64 计算"
                    )
            self._wrapped_model.convert_to_fp64()
        else:
            self._wrapped_model.convert_to_fp32()
        
        self.precision = precision


def create_precision_trainer(
    model: nn.Module,
    optimizer: Optimizer,
    config: Dict[str, Any],
    device: Optional[str] = None
) -> PrecisionTrainer:
    """
    从配置创建精度训练器的工厂函数
    
    Args:
        model: 训练模型
        optimizer: 优化器
        config: 配置字典，应包含 'precision' 键
        device: 训练设备（可选，默认从配置读取）
        
    Returns:
        PrecisionTrainer 实例
    """
    precision = config.get("precision", "fp32")
    device = device or config.get("device", "cuda:0")
    auto_fallback = config.get("auto_fallback", True)
    
    return PrecisionTrainer(
        model=model,
        optimizer=optimizer,
        precision=precision,
        device=device,
        auto_fallback=auto_fallback
    )
