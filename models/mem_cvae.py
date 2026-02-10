import torch
from torch import nn
from models.vunet import VUnet
from models.ml_memAE_sc import ML_MemAE_SC


class BBoxFeatureFusion(nn.Module):
    """
    边界框特征融合模块
    将边界框信息（中心点位置、物体类型）与图像特征融合
    """

    def __init__(self, img_channels=3, bbox_feat_dim=3, fusion_dim=64):
        super(BBoxFeatureFusion, self).__init__()

        self.img_channels = img_channels
        self.bbox_feat_dim = bbox_feat_dim  # [cx, cy, obj_type]
        self.fusion_dim = fusion_dim

        # 图像特征投影
        self.img_proj = nn.Conv2d(img_channels, fusion_dim, 1)

        # 边界框特征处理
        self.bbox_fc = nn.Linear(bbox_feat_dim, fusion_dim)

        # 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim // 4, fusion_dim),
            nn.Sigmoid()
        )

        # 融合后的特征增强
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_dim, img_channels, 1)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, img_features, bbox_features, spatial_size):
        """
        Args:
            img_features: [B, C, H, W] 原始图像特征
            bbox_features: [B, 3] 边界框特征 [cx, cy, obj_type]
            spatial_size: (H, W) 目标空间尺寸
        Returns:
            enhanced_features: [B, C, H, W] 增强后的特征
        """
        batch_size = img_features.size(0)
        H, W = spatial_size

        # 1. 图像特征投影
        img_proj = self.img_proj(img_features)  # [B, fusion_dim, H, W]

        # 2. 边界框特征处理
        bbox_proj = self.bbox_fc(bbox_features)  # [B, fusion_dim]

        # 3. 将边界框特征扩展到空间维度
        bbox_spatial = bbox_proj.unsqueeze(-1).unsqueeze(-1)  # [B, fusion_dim, 1, 1]
        bbox_spatial = bbox_spatial.expand(-1, -1, H, W)  # [B, fusion_dim, H, W]

        # 4. 特征融合
        fused_features = img_proj + bbox_spatial  # [B, fusion_dim, H, W]

        # 5. 通道注意力加权
        # 全局平均池化获取通道权重
        channel_weights = torch.mean(fused_features, dim=[2, 3])  # [B, fusion_dim]
        channel_weights = self.channel_attention(channel_weights)  # [B, fusion_dim]
        channel_weights = channel_weights.unsqueeze(-1).unsqueeze(-1)  # [B, fusion_dim, 1, 1]

        # 应用通道注意力
        weighted_features = fused_features * channel_weights  # [B, fusion_dim, H, W]

        # 6. 特征增强
        enhanced_fusion = self.fusion_conv(weighted_features)  # [B, img_channels, H, W]

        # 7. 残差连接
        final_features = img_features + enhanced_fusion

        return final_features


class HFVAD(nn.Module):
    """
    ML-MemAE-SC + CVAE + 边界框特征融合
    """

    def __init__(self, num_hist, num_pred, config, features_root, num_slots, shrink_thres, skip_ops, mem_usage,
                 finetune=False):
        super(HFVAD, self).__init__()

        self.num_hist = num_hist
        self.num_pred = num_pred
        self.features_root = features_root
        self.num_slots = num_slots
        self.shrink_thres = shrink_thres
        self.skip_ops = skip_ops
        self.mem_usage = mem_usage
        self.finetune = finetune

        self.x_ch = 3  # num of RGB channels
        self.y_ch = 2  # num of optical flow channels

        # 添加边界框特征融合模块
        self.bbox_fusion = BBoxFeatureFusion(
            img_channels=self.x_ch,
            bbox_feat_dim=3,  # [cx, cy, obj_type]
            fusion_dim=64
        )

        self.memAE = ML_MemAE_SC(num_in_ch=self.y_ch, seq_len=1, features_root=self.features_root,
                                 num_slots=self.num_slots, shrink_thres=self.shrink_thres,
                                 mem_usage=self.mem_usage,
                                 skip_ops=self.skip_ops)

        self.vunet = VUnet(config)

        self.mse_loss = nn.MSELoss()

    def forward(self, sample_frame, sample_of, bbox_features=None, mode="train"):
        """
        :param sample_frame: 5 frames in a video clip
        :param sample_of: 4 corresponding flows
        :param bbox_features: [B, 3] 边界框特征 [cx, cy, obj_type]
        :return:
        """
        # 批量处理光流帧，避免逐帧循环调用 memAE
        # 原始实现: 循环 num_hist 次，每次处理 [B, 2, H, W]
        # 优化实现: 一次性处理 [B*num_hist, 2, H, W]，减少前向传播开销
        
        batch_size = sample_of.size(0)
        H, W = sample_of.size(2), sample_of.size(3)
        
        # 将光流帧重组为批量格式: [B, num_hist*2, H, W] -> [B*num_hist, 2, H, W]
        # 首先将通道维度拆分为 [B, num_hist, 2, H, W]
        of_frames = sample_of.view(batch_size, self.num_hist, self.y_ch, H, W)
        # 然后合并 batch 和 num_hist 维度: [B*num_hist, 2, H, W]
        of_batched = of_frames.reshape(batch_size * self.num_hist, self.y_ch, H, W)
        
        # 单次批量前向传播，替代原来的 num_hist 次循环
        memAE_out = self.memAE(of_batched)
        
        # 获取批量输出
        recon_batched = memAE_out["recon"]  # [B*num_hist, 2, H, W]
        att_weight3_batched = memAE_out["att_weight3"]  # [B*num_hist, num_slots]
        att_weight2_batched = memAE_out["att_weight2"]  # [B*num_hist, num_slots]
        att_weight1_batched = memAE_out["att_weight1"]  # [B*num_hist, num_slots]
        
        # 将重建结果恢复为原始格式: [B*num_hist, 2, H, W] -> [B, num_hist*2, H, W]
        of_recon = recon_batched.view(batch_size, self.num_hist, self.y_ch, H, W)
        of_recon = of_recon.reshape(batch_size, self.num_hist * self.y_ch, H, W)
        
        # 注意力权重保持 [B*num_hist, num_slots] 格式，与原始实现的 torch.cat 结果一致
        att_weight3 = att_weight3_batched
        att_weight2 = att_weight2_batched
        att_weight1 = att_weight1_batched

        if self.finetune:
            loss_recon = self.mse_loss(of_recon, sample_of)
            loss_sparsity = torch.mean(
                torch.sum(-att_weight3 * torch.log(att_weight3 + 1e-12), dim=1)
            ) + torch.mean(
                torch.sum(-att_weight2 * torch.log(att_weight2 + 1e-12), dim=1)
            ) + torch.mean(
                torch.sum(-att_weight1 * torch.log(att_weight1 + 1e-12), dim=1)
            )

        frame_in = sample_frame[:, :-self.x_ch * self.num_pred, :, :]
        frame_target = sample_frame[:, -self.x_ch * self.num_pred:, :, :]

        # 应用边界框特征融合
        if bbox_features is not None:
            B, C, H, W = frame_in.shape
            # 将输入帧拆分为单个帧进行处理
            single_frames = []
            for i in range(frame_in.size(1) // self.x_ch):
                frame_slice = frame_in[:, i * self.x_ch:(i + 1) * self.x_ch, :, :]
                # 应用特征融合
                enhanced_frame = self.bbox_fusion(frame_slice, bbox_features, (H, W))
                single_frames.append(enhanced_frame)

            # 重新拼接处理后的帧
            frame_in_enhanced = torch.cat(single_frames, dim=1)
        else:
            frame_in_enhanced = frame_in

        input_dict = dict(appearance=frame_in_enhanced, motion=of_recon)
        frame_pred = self.vunet(input_dict, mode=mode)

        out = dict(frame_pred=frame_pred, frame_target=frame_target,
                   of_recon=of_recon, of_target=sample_of)
        out.update(self.vunet.saved_tensors)

        if self.finetune:
            ML_MemAE_SC_dict = dict(loss_recon=loss_recon, loss_sparsity=loss_sparsity)
            out.update(ML_MemAE_SC_dict)

        return out