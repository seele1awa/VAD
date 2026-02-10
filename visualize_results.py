import argparse
import os
import cv2
import numpy as np
import joblib
import pickle
import yaml
from tqdm import tqdm
from scipy import stats
from collections import deque

METADATA = {
    "ped2": {
        "testing_video_num": 12,
        "testing_frames_cnt": [180, 180, 150, 180, 150, 180, 180, 180, 120, 150, 180, 180]
    },
    "avenue": {
        "testing_video_num": 21,
        "testing_frames_cnt": [1439, 1211, 923, 947, 1007, 1283, 605, 36, 1175, 841,
                               472, 1271, 549, 507, 1001, 740, 426, 294, 248, 273, 76],
    },
    "shanghaitech": {
        "testing_video_num": 107,
        "testing_frames_cnt": [265, 433, 337, 601, 505, 409, 457, 313, 409, 337,
                               337, 457, 577, 313, 529, 193, 289, 289, 265, 241,
                               337, 289, 265, 217, 433, 409, 529, 313, 217, 241,
                               313, 193, 265, 317, 457, 337, 361, 529, 409, 313,
                               385, 457, 481, 457, 433, 385, 241, 553, 937, 865,
                               505, 313, 361, 361, 529, 337, 433, 481, 649, 649,
                               409, 337, 769, 433, 241, 217, 265, 265, 217, 265,
                               409, 385, 481, 457, 313, 601, 241, 481, 313, 337,
                               457, 217, 241, 289, 337, 313, 337, 265, 265, 337,
                               361, 433, 241, 433, 601, 505, 337, 601, 265, 313,
                               241, 289, 361, 385, 217, 337, 265]
    },
}


class TemporalBboxTracker:
    """时序边界框跟踪器，用于平滑边界框和减少闪烁"""
    
    def __init__(self, max_age=10, iou_threshold=0.3):
        """
        Args:
            max_age: 跟踪对象最大存活帧数（无匹配时）
            iou_threshold: IoU阈值，用于匹配边界框
        """
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = []  # 当前活跃的跟踪对象
        self.next_id = 0
        
    def compute_iou(self, bbox1, bbox2):
        """计算两个边界框的IoU"""
        x1_min, y1_min, x1_max, y1_max = bbox1[:4]
        x2_min, y2_min, x2_max, y2_max = bbox2[:4]
        
        # 计算交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # 计算并集
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def update(self, bboxes, scores):
        """
        更新跟踪器
        Args:
            bboxes: 当前帧的边界框 [N, 7] (x1, y1, x2, y2, label, cx, cy)
            scores: 当前帧的异常分数 dict {bbox_idx: score}
        Returns:
            tracked_bboxes: 平滑后的边界框
            tracked_scores: 平滑后的分数
            tracked_ids: 跟踪ID
        """
        # 如果没有检测到边界框，更新现有跟踪的年龄
        if len(bboxes) == 0:
            for track in self.tracks:
                track['age'] += 1
            # 移除过老的跟踪
            self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]
            
            # 返回现有跟踪（使用预测位置）
            tracked_bboxes = []
            tracked_scores = []
            tracked_ids = []
            for track in self.tracks:
                tracked_bboxes.append(track['bbox'])
                tracked_scores.append(track['score'])
                tracked_ids.append(track['id'])
            return tracked_bboxes, tracked_scores, tracked_ids
        
        # 匹配当前检测与现有跟踪
        matched_tracks = set()
        matched_detections = set()
        
        for det_idx, bbox in enumerate(bboxes):
            best_iou = 0
            best_track_idx = -1
            
            for track_idx, track in enumerate(self.tracks):
                if track_idx in matched_tracks:
                    continue
                iou = self.compute_iou(bbox, track['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_idx = track_idx
            
            if best_track_idx >= 0:
                # 更新现有跟踪
                track = self.tracks[best_track_idx]
                
                # 使用指数移动平均平滑边界框
                alpha = 0.7  # 新检测的权重
                track['bbox'][:4] = alpha * bbox[:4] + (1 - alpha) * track['bbox'][:4]
                track['bbox'][4:] = bbox[4:]  # 更新标签和中心点
                
                # 平滑分数
                if det_idx in scores:
                    track['score'] = alpha * scores[det_idx] + (1 - alpha) * track['score']
                
                track['age'] = 0  # 重置年龄
                track['hits'] += 1
                
                matched_tracks.add(best_track_idx)
                matched_detections.add(det_idx)
        
        # 为未匹配的检测创建新跟踪
        for det_idx, bbox in enumerate(bboxes):
            if det_idx not in matched_detections:
                new_track = {
                    'id': self.next_id,
                    'bbox': bbox.copy(),
                    'score': scores.get(det_idx, 0.0),
                    'age': 0,
                    'hits': 1
                }
                self.tracks.append(new_track)
                self.next_id += 1
        
        # 增加未匹配跟踪的年龄
        for track_idx, track in enumerate(self.tracks):
            if track_idx not in matched_tracks:
                track['age'] += 1
        
        # 移除过老的跟踪
        self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]
        
        # 返回所有活跃跟踪（只返回至少被检测到2次的）
        tracked_bboxes = []
        tracked_scores = []
        tracked_ids = []
        for track in self.tracks:
            if track['hits'] >= 2 or track['age'] == 0:  # 新检测或稳定跟踪
                tracked_bboxes.append(track['bbox'])
                tracked_scores.append(track['score'])
                tracked_ids.append(track['id'])
        
        return tracked_bboxes, tracked_scores, tracked_ids


class BayesianAnomalyDetector:
    """基于最小风险的贝叶斯决策器"""
    
    def __init__(self, frame_scores, gt_labels=None, false_alarm_cost=1.0, miss_cost=10.0):
        """
        Args:
            frame_scores: 帧级异常分数
            gt_labels: 真实标签（可选，用于估计先验概率）
            false_alarm_cost: 误报代价（正常判为异常）
            miss_cost: 漏检代价（异常判为正常）
        """
        self.frame_scores = frame_scores
        self.false_alarm_cost = false_alarm_cost
        self.miss_cost = miss_cost
        
        # 估计正常和异常的分布参数
        self._estimate_distributions(gt_labels)
        
    def _estimate_distributions(self, gt_labels):
        """估计正常和异常分数的分布"""
        if gt_labels is not None:
            # 如果有真实标签，使用标签分离数据
            normal_scores = self.frame_scores[gt_labels == 0]
            abnormal_scores = self.frame_scores[gt_labels == 1]
            
            if len(normal_scores) > 0:
                self.normal_mean = np.mean(normal_scores)
                self.normal_std = np.std(normal_scores) + 1e-6
            else:
                self.normal_mean = np.min(self.frame_scores)
                self.normal_std = 1.0
                
            if len(abnormal_scores) > 0:
                self.abnormal_mean = np.mean(abnormal_scores)
                self.abnormal_std = np.std(abnormal_scores) + 1e-6
            else:
                self.abnormal_mean = np.max(self.frame_scores)
                self.abnormal_std = 1.0
                
            self.prior_normal = len(normal_scores) / len(self.frame_scores)
            self.prior_abnormal = len(abnormal_scores) / len(self.frame_scores)
        else:
            # 无标签情况：假设低分数为正常，高分数为异常
            threshold_percentile = 80  # 假设80%为正常
            threshold = np.percentile(self.frame_scores, threshold_percentile)
            
            normal_scores = self.frame_scores[self.frame_scores < threshold]
            abnormal_scores = self.frame_scores[self.frame_scores >= threshold]
            
            self.normal_mean = np.mean(normal_scores)
            self.normal_std = np.std(normal_scores) + 1e-6
            self.abnormal_mean = np.mean(abnormal_scores)
            self.abnormal_std = np.std(abnormal_scores) + 1e-6
            
            self.prior_normal = threshold_percentile / 100.0
            self.prior_abnormal = 1 - self.prior_normal
    
    def compute_risk(self, score):
        """计算将某个分数判为正常和异常的风险"""
        # 计算似然概率 P(score|normal) 和 P(score|abnormal)
        likelihood_normal = stats.norm.pdf(score, self.normal_mean, self.normal_std)
        likelihood_abnormal = stats.norm.pdf(score, self.abnormal_mean, self.abnormal_std)
        
        # 计算后验概率 P(normal|score) 和 P(abnormal|score)
        evidence = likelihood_normal * self.prior_normal + likelihood_abnormal * self.prior_abnormal
        if evidence < 1e-10:
            evidence = 1e-10
            
        posterior_normal = (likelihood_normal * self.prior_normal) / evidence
        posterior_abnormal = (likelihood_abnormal * self.prior_abnormal) / evidence
        
        # 计算风险
        # 判为正常的风险 = P(abnormal|score) * miss_cost
        risk_decide_normal = posterior_abnormal * self.miss_cost
        # 判为异常的风险 = P(normal|score) * false_alarm_cost
        risk_decide_abnormal = posterior_normal * self.false_alarm_cost
        
        return risk_decide_normal, risk_decide_abnormal, posterior_abnormal
    
    def predict(self, score):
        """基于最小风险决策"""
        risk_normal, risk_abnormal, prob_abnormal = self.compute_risk(score)
        
        # 选择风险最小的决策
        is_abnormal = risk_abnormal < risk_normal
        
        return is_abnormal, prob_abnormal


def draw_progress_bar(frame, current_frame, total_frames, is_abnormal, bar_height=30):
    """绘制进度条"""
    h, w = frame.shape[:2]
    bar_width = w
    
    # 创建进度条背景
    progress_ratio = current_frame / total_frames
    filled_width = int(bar_width * progress_ratio)
    
    # 根据异常状态选择颜色
    if is_abnormal:
        color = (0, 0, 255)  # 红色 (BGR)
    else:
        color = (0, 255, 0)  # 绿色 (BGR)
    
    # 绘制背景（灰色）
    cv2.rectangle(frame, (0, h - bar_height), (bar_width, h), (100, 100, 100), -1)
    
    # 绘制进度
    cv2.rectangle(frame, (0, h - bar_height), (filled_width, h), color, -1)
    
    # 添加文字
    status_text = "ABNORMAL" if is_abnormal else "NORMAL"
    cv2.putText(frame, f"Frame {current_frame}/{total_frames} - {status_text}", 
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame


def visualize_video(video_id, dataset_name, dataset_base_dir, frame_scores, frame_bbox_scores,
                    bboxes_dict, output_dir, detector, start_frame_idx):
    """可视化单个视频"""
    
    # 获取视频路径
    if dataset_name == "ped2":
        video_dir = os.path.join(dataset_base_dir, dataset_name, "testing", "frames", f"{video_id+1:02d}")
    elif dataset_name == "avenue":
        video_dir = os.path.join(dataset_base_dir, dataset_name, "testing", "frames", f"{video_id+1:02d}")
    elif dataset_name == "shanghaitech":
        video_dir = os.path.join(dataset_base_dir, dataset_name, "testing", "frames", f"{video_id+1:02d}")
    
    if not os.path.exists(video_dir):
        print(f"Warning: Video directory not found: {video_dir}")
        return
    
    # 获取所有帧
    frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith(('.jpg', '.png', '.bmp'))])
    total_frames = len(frame_files)
    
    if total_frames == 0:
        print(f"Warning: No frames found in {video_dir}")
        return
    
    # 读取第一帧获取尺寸
    first_frame = cv2.imread(os.path.join(video_dir, frame_files[0]))
    h, w = first_frame.shape[:2]
    
    # 创建视频写入器
    output_path = os.path.join(output_dir, f"video_{video_id+1:02d}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 25.0, (w, h + 30))  # 增加高度以容纳进度条
    
    # 创建时序跟踪器
    tracker = TemporalBboxTracker(max_age=10, iou_threshold=0.3)
    
    print(f"Processing video {video_id+1}/{METADATA[dataset_name]['testing_video_num']}")
    
    for frame_idx, frame_file in enumerate(tqdm(frame_files, desc=f"Video {video_id+1}")):
        # 读取帧
        frame = cv2.imread(os.path.join(video_dir, frame_file))
        if frame is None:
            continue
        
        # 全局帧索引
        global_frame_idx = start_frame_idx + frame_idx
        
        # 获取该帧的异常分数
        if global_frame_idx < len(frame_scores):
            frame_score = frame_scores[global_frame_idx]
            is_abnormal, prob_abnormal = detector.predict(frame_score)
        else:
            frame_score = 0
            is_abnormal = False
            prob_abnormal = 0
        
        # 获取当前帧的边界框和分数
        current_bboxes = []
        current_scores = {}
        
        if global_frame_idx in bboxes_dict:
            current_bboxes = bboxes_dict[global_frame_idx]
            if global_frame_idx < len(frame_bbox_scores):
                current_scores = frame_bbox_scores[global_frame_idx]
        
        # 使用跟踪器平滑边界框
        tracked_bboxes, tracked_scores, tracked_ids = tracker.update(current_bboxes, current_scores)
        
        # 绘制跟踪后的边界框
        for bbox, score, track_id in zip(tracked_bboxes, tracked_scores, tracked_ids):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # 根据bbox分数决定颜色
            bbox_is_abnormal, bbox_prob = detector.predict(score)
            bbox_color = (0, 0, 255) if bbox_is_abnormal else (0, 255, 0)
            
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
            
            # 绘制分数和ID
            score_text = f"ID:{track_id} {score:.2f} ({bbox_prob*100:.1f}%)"
            cv2.putText(frame, score_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)
        
        # 添加帧级信息
        info_text = f"Frame Score: {frame_score:.2f} | Prob: {prob_abnormal*100:.1f}%"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 扩展帧以容纳进度条
        frame_with_bar = np.zeros((h + 30, w, 3), dtype=np.uint8)
        frame_with_bar[:h, :, :] = frame
        
        # 绘制进度条
        frame_with_bar = draw_progress_bar(frame_with_bar, frame_idx + 1, total_frames, is_abnormal)
        
        # 写入视频
        out.write(frame_with_bar)
    
    out.release()
    print(f"Saved: {output_path}")


def main(args):
    # 加载配置
    config = yaml.safe_load(open(args.cfg_file))
    dataset_name = config["dataset_name"]
    dataset_base_dir = config["dataset_base_dir"]
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, config["exp_name"])
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载评估结果
    eval_dir = os.path.join(config["eval_root"], config["exp_name"])
    frame_scores = joblib.load(os.path.join(eval_dir, f"frame_scores_{args.suffix}.json"))
    print(f"Loaded frame_scores: {len(frame_scores)} frames")
    
    # 尝试加载frame_bbox_scores（如果存在）
    frame_bbox_scores_path = os.path.join(eval_dir, f"frame_bbox_scores_{args.suffix}.json")
    if os.path.exists(frame_bbox_scores_path):
        frame_bbox_scores = joblib.load(frame_bbox_scores_path)
        print(f"Loaded frame_bbox_scores: {len(frame_bbox_scores)} frames")
    else:
        print("Warning: frame_bbox_scores not found, will only show frame-level scores")
        frame_bbox_scores = [{} for _ in range(len(frame_scores))]
    
    # 加载边界框
    # extract_bboxes.py 保存的格式是 {dataset_name}_bboxes_test.npy
    bboxes_file = os.path.join(dataset_base_dir, dataset_name, f"{dataset_name}_bboxes_test.npy")
    
    if os.path.exists(bboxes_file):
        all_bboxes_array = np.load(bboxes_file, allow_pickle=True)
        print(f"Loaded bboxes from: {bboxes_file}")
        print(f"Total frames with bboxes: {len(all_bboxes_array)}")
        
        # 转换为字典格式 {frame_idx: bboxes}
        all_bboxes = {}
        for frame_idx, bboxes in enumerate(all_bboxes_array):
            all_bboxes[frame_idx] = bboxes
    else:
        print(f"Warning: Bboxes file not found: {bboxes_file}")
        print("Will only show frame-level scores without bounding boxes")
        all_bboxes = {}
    
    # 加载真实标签（用于贝叶斯决策器）
    gt_file = os.path.join(dataset_base_dir, dataset_name, "ground_truth_demo", "gt_label.json")
    if os.path.exists(gt_file):
        gt = pickle.load(open(gt_file, "rb"))
        gt_concat = np.concatenate(list(gt.values()), axis=0)
        print(f"Loaded ground truth: {len(gt_concat)} frames (before removing first 4 frames)")
        
        # 去除前4帧（与frame_scores对齐）
        new_gt = np.array([])
        start_idx = 0
        for cur_video_id in range(METADATA[dataset_name]["testing_video_num"]):
            gt_each_video = gt_concat[start_idx:start_idx + METADATA[dataset_name]["testing_frames_cnt"][cur_video_id]][4:]
            start_idx += METADATA[dataset_name]["testing_frames_cnt"][cur_video_id]
            new_gt = np.concatenate((new_gt, gt_each_video), axis=0)
        gt_labels = new_gt
        print(f"Ground truth after removing first 4 frames: {len(gt_labels)} frames")
        
        # 确保长度匹配
        if len(gt_labels) != len(frame_scores):
            print(f"Warning: Length mismatch - gt_labels: {len(gt_labels)}, frame_scores: {len(frame_scores)}")
            print("Using unsupervised estimation instead")
            gt_labels = None
    else:
        print("Warning: Ground truth not found, using unsupervised estimation")
        gt_labels = None
    
    # 创建贝叶斯决策器
    detector = BayesianAnomalyDetector(
        frame_scores, 
        gt_labels=gt_labels,
        false_alarm_cost=args.false_alarm_cost,
        miss_cost=args.miss_cost
    )
    
    print(f"Bayesian Detector Parameters:")
    print(f"  Normal: mean={detector.normal_mean:.3f}, std={detector.normal_std:.3f}")
    print(f"  Abnormal: mean={detector.abnormal_mean:.3f}, std={detector.abnormal_std:.3f}")
    print(f"  Prior: P(normal)={detector.prior_normal:.3f}, P(abnormal)={detector.prior_abnormal:.3f}")
    print(f"  Costs: False Alarm={args.false_alarm_cost}, Miss={args.miss_cost}")
    
    # 处理每个视频
    start_frame_idx = 0
    for video_id in range(METADATA[dataset_name]["testing_video_num"]):
        visualize_video(
            video_id, 
            dataset_name, 
            dataset_base_dir, 
            frame_scores, 
            frame_bbox_scores,
            all_bboxes, 
            output_dir, 
            detector,
            start_frame_idx
        )
        start_frame_idx += METADATA[dataset_name]["testing_frames_cnt"][video_id]
    
    print(f"\nAll videos saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize anomaly detection results with Bayesian decision')
    parser.add_argument("--cfg_file", type=str, required=True,
                        help='path to model config file')
    parser.add_argument("--output_dir", type=str, default="./visualizations",
                        help='directory to save output videos')
    parser.add_argument("--suffix", type=str, default="best",
                        help='suffix for loading evaluation results')
    parser.add_argument("--false_alarm_cost", type=float, default=1.0,
                        help='cost of false alarm (normal classified as abnormal)')
    parser.add_argument("--miss_cost", type=float, default=10.0,
                        help='cost of miss detection (abnormal classified as normal)')
    
    args = parser.parse_args()
    main(args)
