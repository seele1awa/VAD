import numpy as np
import os
import argparse
import cv2
import torch
from tqdm import tqdm
from datasets.dataset import get_dataset, img_tensor2numpy, img_batch_tensor2numpy
# 使用官方 mmdet API 替代自定义的 mmdet_utils
from mmdet.apis import init_detector, inference_detector

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

DATASET_CFGS = {
    "ped2": {"conf_thr": 0.5, "min_area": 10 * 10, "cover_thr": 0.6, "binary_thr": 18, "gauss_mask_size": 3,
             'contour_min_area': 10 * 10},
    "avenue": {"conf_thr": 0.25, "min_area": 40 * 40, "cover_thr": 0.6, "binary_thr": 18, "gauss_mask_size": 5,
               'contour_min_area': 40 * 40},
    "shanghaitech": {"conf_thr": 0.5, "min_area": 8 * 8, "cover_thr": 0.65, "binary_thr": 15, "gauss_mask_size": 5,
                     'contour_min_area': 40 * 40},
    "cusco": {"conf_thr": 0.5, "min_area": 10 * 10, "cover_thr": 0.6, "binary_thr": 18, "gauss_mask_size": 3,
              'contour_min_area': 10 * 10}
}


def getObjBboxes(img, model, dataset_name, img_width, img_height):
    # 使用官方 inference_detector
    result = inference_detector(model, img)
    CONF_THR = DATASET_CFGS[dataset_name]["conf_thr"]
    MIN_AREA = DATASET_CFGS[dataset_name]["min_area"]

    # 新版 mmdet 返回的是 DetDataSample 对象
    # 我们需要从中提取检测结果
    if hasattr(result, 'pred_instances'):
        # 单个 DetDataSample 的情况
        pred_instances = result.pred_instances
        bboxes = pred_instances.bboxes.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()
    elif isinstance(result, list) and len(result) > 0 and hasattr(result[0], 'pred_instances'):
        # 列表包含多个 DetDataSample 的情况
        pred_instances = result[0].pred_instances
        bboxes = pred_instances.bboxes.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()
    else:
        # 如果格式不符合预期，返回空数组
        print('fuck1')
        return np.empty((0, 7))  # 修改为7列

    # 将检测结果组合成 [x1, y1, x2, y2, class_idx, score, cx, cy] 的格式
    bboxes_list = []
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        score = scores[i]
        label = labels[i]
        
        # 计算中心点坐标
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        
        # 归一化中心点坐标
        cx_norm = cx / img_width
        cy_norm = cy / img_height
        
        bboxes_list.append([bbox[0], bbox[1], bbox[2], bbox[3], label, score, cx_norm, cy_norm])

    if bboxes_list:
        bboxes = np.array(bboxes_list)
    else:
        print('fuck2')
        bboxes = np.empty((0, 8))  # 修改为8列

    scores = bboxes[:, 5]  # x1,y1,x2,y2,label,score,cx,cy
    bboxes = bboxes[scores > CONF_THR, :]

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    bbox_areas = (y2 - y1 + 1) * (x2 - x1 + 1)

    # 修复索引问题：先选择行，再选择列
    area_mask = bbox_areas >= MIN_AREA
    valid_bboxes = bboxes[area_mask]
    
    # 返回 [x1, y1, x2, y2, label, cx_norm, cy_norm] - 去掉score列
    if valid_bboxes.size > 0:
        valid_bboxes = valid_bboxes[:, [0, 1, 2, 3, 4, 6, 7]]
    else:
        valid_bboxes = np.empty((0, 7))
        
    return valid_bboxes


def delCoverBboxes(bboxes, dataset_name):
    assert bboxes.ndim == 2
    assert bboxes.shape[1] == 7  # 修改为7列

    COVER_THR = DATASET_CFGS[dataset_name]["cover_thr"]

    # 只使用前4列进行重叠计算（x1,y1,x2,y2）
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    bbox_areas = (y2 - y1 + 1) * (x2 - x1 + 1)

    sort_idx = bbox_areas.argsort()  # Index of bboxes sorted in ascending order by area size

    keep_idx = []
    for i in range(sort_idx.size):  # calculate overlap with i-th bbox
        # Calculate the point coordinates of the intersection
        x11 = np.maximum(x1[sort_idx[i]], x1[sort_idx[i + 1:]])
        y11 = np.maximum(y1[sort_idx[i]], y1[sort_idx[i + 1:]])
        x22 = np.minimum(x2[sort_idx[i]], x2[sort_idx[i + 1:]])
        y22 = np.minimum(y2[sort_idx[i]], y2[sort_idx[i + 1:]])
        # Calculate the intersection area
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        overlaps = w * h

        ratios = overlaps / bbox_areas[sort_idx[i]]
        num = ratios[ratios > COVER_THR]
        if num.size == 0:
            keep_idx.append(sort_idx[i])

    return bboxes[keep_idx]


def getFgBboxes(cur_img, img_batch, bboxes, dataset_name):
    area_thr = DATASET_CFGS[dataset_name]["contour_min_area"]
    binary_thr = DATASET_CFGS[dataset_name]["binary_thr"]
    gauss_mask_size = DATASET_CFGS[dataset_name]["gauss_mask_size"]
    extend = 2

    sum_grad = 0
    for i in range(img_batch.shape[0] - 1):
        img1 = img_batch[i, :, :, :]
        img2 = img_batch[i + 1, :, :, :]
        img1 = cv2.GaussianBlur(img1, (gauss_mask_size, gauss_mask_size), 0)
        img2 = cv2.GaussianBlur(img2, (gauss_mask_size, gauss_mask_size), 0)

        grad = cv2.absdiff(img1, img2)
        sum_grad = grad + sum_grad

    sum_grad = cv2.threshold(sum_grad, binary_thr, 255, cv2.THRESH_BINARY)[1]  # temporal gradient

    # 只使用前4列进行前景检测（x1,y1,x2,y2）
    for bbox in bboxes:
        bbox_int = bbox[:4].astype(np.int32)  # 只取前4个元素
        extend_y1 = np.maximum(0, bbox_int[1] - extend)
        extend_y2 = np.minimum(bbox_int[3] + extend, sum_grad.shape[0])
        extend_x1 = np.maximum(0, bbox_int[0] - extend)
        extend_x2 = np.minimum(bbox_int[2] + extend, sum_grad.shape[1])
        sum_grad[extend_y1:extend_y2 + 1, extend_x1:extend_x2 + 1] = 0

    sum_grad = cv2.cvtColor(sum_grad, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(sum_grad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fg_bboxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        sum_grad = cv2.rectangle(sum_grad, (x, y), (x + w, y + h), color=255, thickness=1)
        area = (w + 1) * (h + 1)
        if area > area_thr and w / h < 10 and h / w < 10:
            extend_x1 = np.maximum(0, x - extend)
            extend_y1 = np.maximum(0, y - extend)
            extend_x2 = np.minimum(x + w + extend, sum_grad.shape[1])
            extend_y2 = np.minimum(y + h + extend, sum_grad.shape[0])
            # 对于前景边界框，设置默认的类别和中心点
            cx_norm = (extend_x1 + extend_x2) / 2.0 / cur_img.shape[1]
            cy_norm = (extend_y1 + extend_y2) / 2.0 / cur_img.shape[0]
            fg_bboxes.append([extend_x1, extend_y1, extend_x2, extend_y2, 0, cx_norm, cy_norm])  # 类别设为0

    return np.array(fg_bboxes)


def obj_bboxes_extraction(dataset_root, dataset_name, mode):
    # mmdet config file and pre-trained model weights
    mm_det_config_file = 'assets/yolox_l_8xb8-300e_coco.py'
    mm_det_ckpt_file = 'assets/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'

    dataset = get_dataset(dataset_name=dataset_name,
                          dir=os.path.join(dataset_root, dataset_name),
                          context_frame_num=1, mode=mode)

    # 使用官方 init_detector
    mm_det_model = init_detector(mm_det_config_file, mm_det_ckpt_file, device="cuda:0")

    all_bboxes = list()

    for idx in tqdm(range(len(dataset)), total=len(dataset)):
        batch, _ = dataset.__getitem__(idx)

        # centric frame
        cur_img = img_tensor2numpy(batch[1])
        h, w = cur_img.shape[0], cur_img.shape[1]

        obj_bboxes = getObjBboxes(cur_img, mm_det_model, dataset_name, w, h)
        # filter some overlapped bbox
        obj_bboxes = delCoverBboxes(obj_bboxes, dataset_name)
        fg_bboxes = getFgBboxes(cur_img, img_batch_tensor2numpy(batch), obj_bboxes, dataset_name)

        # 确保所有边界框数组都是二维的
        if obj_bboxes.ndim == 1 and obj_bboxes.size > 0:
            obj_bboxes = obj_bboxes.reshape(1, -1)
        elif obj_bboxes.size == 0:
            obj_bboxes = np.empty((0, 7))  # 修改为7列

        if fg_bboxes.ndim == 1 and fg_bboxes.size > 0:
            fg_bboxes = fg_bboxes.reshape(1, -1)
        elif fg_bboxes.size == 0:
            fg_bboxes = np.empty((0, 7))  # 修改为7列

        if fg_bboxes.shape[0] > 0 and obj_bboxes.shape[0] > 0:
            cur_bboxes = np.concatenate((obj_bboxes, fg_bboxes), axis=0)
        elif fg_bboxes.shape[0] > 0:
            cur_bboxes = fg_bboxes
        elif obj_bboxes.shape[0] > 0:
            cur_bboxes = obj_bboxes
        else:
            # 如果没有检测到任何边界框，创建一个空的二维数组
            cur_bboxes = np.empty((0, 7))  # 修改为7列

        all_bboxes.append(cur_bboxes)

    # 保存为对象数组
    all_bboxes_array = np.empty(len(all_bboxes), dtype=object)
    for i, bboxes in enumerate(all_bboxes):
        all_bboxes_array[i] = bboxes

    np.save(os.path.join(os.path.join(dataset_root, dataset_name),
                         '%s_bboxes_%s.npy' % (dataset_name, mode)), all_bboxes_array)
    print('bboxes saved!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_root", type=str, default="/home/liuzhian/hdd4T/code/hf2vad", help='project root path')
    parser.add_argument("--dataset_name", type=str, default="ped2", help='dataset name')
    parser.add_argument("--mode", type=str, default="train", help='train or test data')
    args = parser.parse_args()

    obj_bboxes_extraction(dataset_root=os.path.join(args.proj_root, "data"),
                          dataset_name=args.dataset_name,
                          mode=args.mode)