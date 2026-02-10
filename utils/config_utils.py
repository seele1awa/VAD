"""
配置管理模块

提供配置加载、继承和验证的统一工具。
支持从 YAML 文件加载配置，并实现配置继承机制。
"""

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import copy


@dataclass
class ModelConfig:
    """
    模型配置数据类
    
    Attributes:
        version: 模型版本 ('baseline', 'v2', 'v3', etc.)
        num_slots: 记忆槽数量
        shrink_thres: 收缩阈值
        features_root: 特征根数量
        mem_usage: 各层是否使用记忆模块
        skip_ops: 跳跃连接操作类型
        
        # Version A specific - Memory模块改进
        learnable_threshold: 是否使用可学习阈值
        num_attention_heads: 注意力头数
        
        # Version B specific - VUnet注意力机制
        use_cbam: 是否使用CBAM模块
        use_se_block: 是否使用SE Block
        
        # Version C specific - 损失函数改进
        use_perceptual_loss: 是否使用感知损失
        use_ssim_loss: 是否使用SSIM损失
        perceptual_weight: 感知损失权重
        ssim_weight: SSIM损失权重
        
        # Version D specific - 边界框融合改进
        use_spatial_attention: 是否使用空间注意力
        use_cross_attention: 是否使用交叉注意力
        
        # Version E specific - 多尺度特征融合
        use_fpn: 是否使用FPN
        
        # Version F specific - 时序建模增强
        use_3d_conv: 是否使用3D卷积
        use_temporal_transformer: 是否使用时序Transformer
    """
    version: str = "baseline"
    num_slots: int = 2000
    shrink_thres: float = 0.0005
    features_root: int = 32
    mem_usage: List[bool] = field(default_factory=lambda: [False, True, True, True])
    skip_ops: List[str] = field(default_factory=lambda: ["none", "concat", "concat"])
    
    # Version A specific
    learnable_threshold: bool = False
    num_attention_heads: int = 4
    
    # Version B specific
    use_cbam: bool = False
    use_se_block: bool = False
    
    # Version C specific
    use_perceptual_loss: bool = False
    use_ssim_loss: bool = False
    perceptual_weight: float = 0.1
    ssim_weight: float = 0.1

    # Version D specific
    use_spatial_attention: bool = False
    use_cross_attention: bool = False
    
    # Version E specific
    use_fpn: bool = False
    
    # Version F specific
    use_3d_conv: bool = False
    use_temporal_transformer: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """从字典创建配置"""
        # 只保留有效的字段
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass
class TrainingConfig:
    """
    训练配置数据类
    
    Attributes:
        batch_size: 批次大小
        num_epochs: 训练轮数
        learning_rate: 学习率
        num_workers: 数据加载工作进程数
        precision: 训练精度 ('fp32' 或 'fp64')
        gradient_accumulation_steps: 梯度累积步数
        device: 训练设备
        
        # 损失函数权重
        lam_frame: 帧重建损失权重
        lam_kl: KL散度损失权重
        lam_grad: 梯度损失权重
        lam_sparse: 稀疏损失权重
        lam_recon: 重建损失权重
        
        # 评估权重
        w_r: 光流重建权重
        w_p: 帧预测权重
    """
    batch_size: int = 128
    num_epochs: int = 80
    learning_rate: float = 0.0001
    num_workers: int = 8
    precision: str = "fp32"
    gradient_accumulation_steps: int = 1
    device: str = "cuda:0"
    
    # 损失函数权重
    lam_frame: float = 1.0
    lam_kl: float = 1.0
    lam_grad: float = 1.0
    lam_sparse: float = 0.0002
    lam_recon: float = 1.0
    alpha: int = 1  # gradient loss alpha
    intensity_loss_norm: int = 2
    
    # 评估权重
    w_r: float = 1.0
    w_p: float = 0.1
    
    # 日志和保存设置
    logevery: int = 100
    saveevery: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """从字典创建配置"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass
class AblationConfig:
    """
    消融实验配置数据类
    
    Attributes:
        experiment_name: 实验名称
        model_versions: 要测试的模型版本列表
        num_runs: 每个版本运行次数
        save_checkpoints: 是否保存检查点
        generate_plots: 是否生成可视化图表
        base_config: 基础配置文件路径
    """
    experiment_name: str = "ablation_study"
    model_versions: List[str] = field(default_factory=lambda: ["baseline"])
    num_runs: int = 3
    save_checkpoints: bool = True
    generate_plots: bool = True
    base_config: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AblationConfig":
        """从字典创建配置"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并两个字典，override中的值会覆盖base中的值
    
    Args:
        base: 基础字典
        override: 覆盖字典
        
    Returns:
        合并后的字典
        
    Example:
        >>> base = {'a': 1, 'b': {'c': 2, 'd': 3}}
        >>> override = {'b': {'c': 4}}
        >>> deep_merge(base, override)
        {'a': 1, 'b': {'c': 4, 'd': 3}}
    """
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    
    return result


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    从YAML文件加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
        
    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML解析错误
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config if config is not None else {}


def save_yaml_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    保存配置到YAML文件
    
    Args:
        config: 配置字典
        config_path: 配置文件路径
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def load_config_with_inheritance(
    config_path: Union[str, Path],
    base_key: str = "_base_"
) -> Dict[str, Any]:
    """
    加载配置文件，支持配置继承
    
    配置文件可以通过 _base_ 键指定父配置文件，
    子配置会继承父配置的所有值，并可以覆盖特定值。
    
    Args:
        config_path: 配置文件路径
        base_key: 指定父配置的键名，默认为 "_base_"
        
    Returns:
        合并后的配置字典
        
    Example:
        # child_config.yaml
        _base_: parent_config.yaml
        model_paras:
          num_slots: 3000  # 覆盖父配置的值
          
        # 加载后会继承parent_config.yaml的所有值，
        # 并用child_config.yaml中的值覆盖
    """
    config_path = Path(config_path)
    config = load_yaml_config(config_path)
    
    # 检查是否有父配置
    if base_key in config:
        base_path = config.pop(base_key)
        
        # 处理相对路径
        if not os.path.isabs(base_path):
            base_path = config_path.parent / base_path
        
        # 递归加载父配置
        base_config = load_config_with_inheritance(base_path, base_key)
        
        # 合并配置
        config = deep_merge(base_config, config)
    
    return config


class ConfigManager:
    """
    配置管理器
    
    提供配置加载、验证和管理的统一接口。
    
    Attributes:
        config: 原始配置字典
        model_config: 模型配置对象
        training_config: 训练配置对象
        
    Example:
        >>> manager = ConfigManager.from_yaml("cfgs/cfg.yaml")
        >>> print(manager.model_config.num_slots)
        2000
        >>> print(manager.training_config.batch_size)
        128
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None
    ):
        self.config = config
        self._model_config = model_config
        self._training_config = training_config
    
    @classmethod
    def from_yaml(
        cls,
        config_path: Union[str, Path],
        use_inheritance: bool = True
    ) -> "ConfigManager":
        """
        从YAML文件创建配置管理器
        
        Args:
            config_path: 配置文件路径
            use_inheritance: 是否使用配置继承
            
        Returns:
            ConfigManager实例
        """
        if use_inheritance:
            config = load_config_with_inheritance(config_path)
        else:
            config = load_yaml_config(config_path)
        
        return cls(config)
    
    @property
    def model_config(self) -> ModelConfig:
        """获取模型配置"""
        if self._model_config is None:
            model_paras = self.config.get("model_paras", {})
            # 映射配置键名
            mapped_config = {
                "num_slots": model_paras.get("num_slots", 2000),
                "shrink_thres": model_paras.get("shrink_thres", 0.0005),
                "features_root": model_paras.get("feature_root", 32),
                "mem_usage": model_paras.get("mem_usage", [False, True, True, True]),
                "skip_ops": model_paras.get("skip_ops", ["none", "concat", "concat"]),
            }
            # 添加版本特定配置
            for key in ["version", "learnable_threshold", "num_attention_heads",
                       "use_cbam", "use_se_block", "use_perceptual_loss",
                       "use_ssim_loss", "perceptual_weight", "ssim_weight",
                       "use_spatial_attention", "use_cross_attention",
                       "use_fpn", "use_3d_conv", "use_temporal_transformer"]:
                if key in model_paras:
                    mapped_config[key] = model_paras[key]
            
            self._model_config = ModelConfig.from_dict(mapped_config)
        return self._model_config
    
    @property
    def training_config(self) -> TrainingConfig:
        """获取训练配置"""
        if self._training_config is None:
            # 映射配置键名
            mapped_config = {
                "batch_size": self.config.get("batchsize", 128),
                "num_epochs": self.config.get("num_epochs", 80),
                "learning_rate": self.config.get("lr", 0.0001),
                "num_workers": self.config.get("num_workers", 8),
                "device": self.config.get("device", "cuda:0"),
                "precision": self.config.get("precision", "fp32"),
                "lam_frame": self.config.get("lam_frame", 1.0),
                "lam_kl": self.config.get("lam_kl", 1.0),
                "lam_grad": self.config.get("lam_grad", 1.0),
                "lam_sparse": self.config.get("lam_sparse", 0.0002),
                "lam_recon": self.config.get("lam_recon", 1.0),
                "alpha": self.config.get("alpha", 1),
                "intensity_loss_norm": self.config.get("intensity_loss_norm", 2),
                "w_r": self.config.get("w_r", 1.0),
                "w_p": self.config.get("w_p", 0.1),
                "logevery": self.config.get("logevery", 100),
                "saveevery": self.config.get("saveevery", 1),
            }
            self._training_config = TrainingConfig.from_dict(mapped_config)
        return self._training_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """通过索引获取配置值"""
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """检查配置键是否存在"""
        return key in self.config
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return copy.deepcopy(self.config)
    
    def save(self, config_path: Union[str, Path]) -> None:
        """保存配置到文件"""
        save_yaml_config(self.config, config_path)


def create_ablation_config(
    base_config_path: Union[str, Path],
    version: str,
    output_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    创建消融实验配置
    
    根据基础配置和版本创建特定的消融实验配置。
    
    Args:
        base_config_path: 基础配置文件路径
        version: 模型版本 ('baseline', 'version_a', 'version_b', etc.)
        output_path: 输出配置文件路径（可选）
        
    Returns:
        消融实验配置字典
    """
    config = load_config_with_inheritance(base_config_path)
    
    # 根据版本设置特定配置
    version_configs = {
        "baseline": {},
        "version_a": {
            "model_paras": {
                "version": "v2",
                "learnable_threshold": True,
                "num_attention_heads": 4,
            }
        },
        "version_b": {
            "model_paras": {
                "version": "v2",
                "use_cbam": True,
                "use_se_block": True,
            }
        },
        "version_c": {
            "model_paras": {
                "use_perceptual_loss": True,
                "use_ssim_loss": True,
                "perceptual_weight": 0.1,
                "ssim_weight": 0.1,
            }
        },
        "version_d": {
            "model_paras": {
                "version": "v2",
                "use_spatial_attention": True,
                "use_cross_attention": True,
            }
        },
        "version_e": {
            "model_paras": {
                "version": "v3",
                "use_fpn": True,
            }
        },
        "version_f": {
            "model_paras": {
                "version": "v3",
                "use_3d_conv": True,
                "use_temporal_transformer": True,
            }
        },
    }
    
    if version in version_configs:
        config = deep_merge(config, version_configs[version])
    
    # 更新实验名称
    config["exp_name"] = f"{config.get('exp_name', 'experiment')}_{version}"
    
    if output_path:
        save_yaml_config(config, output_path)
    
    return config
