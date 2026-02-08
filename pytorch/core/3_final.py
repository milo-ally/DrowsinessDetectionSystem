# -*- coding: UTF-8 -*-
import cv2
import torch
import torch.nn as nn
import numpy as np
import math
from pathlib import Path
import sys
import torchvision
from PIL import Image 
import time
import torchvision.transforms as T

# -------------------------- 人脸检测模型相关函数 --------------------------
def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Ensemble(nn.ModuleList):
    def __init__(self):
        super().__init__()
    def forward(self, x, augment=False):
        y = [module(x, augment)[0] for module in self]
        y = torch.cat(y, 1)
        return y, None

# 模型加载
def attempt_load(weights, map_location=None):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        model.append(torch.load(w, map_location=map_location, weights_only=False)['model'].float().fuse().eval())
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()
    return model[-1] if len(model) == 1 else model

# 图像预处理：letterbox缩放（保持比例+补边）
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)	

# 坐标转换/IOU/NMS
def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

# 计算IOU
def box_iou(box1, box2):
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])
    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)

# 非极大值抑制NMS
def non_max_suppression_face(prediction, conf_thres=0.25, iou_thres=0.45):
    nc = prediction.shape[2] - 15
    xc = prediction[..., 4] > conf_thres
    min_wh, max_wh = 2, 4096
    output = [torch.zeros((0, 16), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        x[:, 15:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        conf, j = x[:, 15:].max(1, keepdim=True)
        x = torch.cat((box, conf, x[:, 5:15], j.float()), 1)[conf.view(-1) > conf_thres]
        n = x.shape[0]
        if not n:
            continue
        c = x[:, 15:16] * 0
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        output[xi] = x[i]
    return output

# 坐标缩放（框+关键点）
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    return coords

# 关键点坐标缩放
def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2, 4, 6, 8]] -= pad[0]
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]
    coords[:, :10] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    coords[:, 4].clamp_(0, img0_shape[1])
    coords[:, 5].clamp_(0, img0_shape[0])
    coords[:, 6].clamp_(0, img0_shape[1])
    coords[:, 7].clamp_(0, img0_shape[0])
    coords[:, 8].clamp_(0, img0_shape[1])
    coords[:, 9].clamp_(0, img0_shape[0])
    return coords

# 预处理函数（适配摄像头帧）
def preprocess_image(img, device, img_size=640):
    img0 = img.copy()
    img_letter, ratio, pad = letterbox(img0, new_shape=img_size)
    img_tensor = img_letter[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    img_tensor = np.ascontiguousarray(img_tensor) # 转为连续数组 
    img_tensor = torch.from_numpy(img_tensor).to(device) # 转换为tensor并移动到指定设备
    img_tensor = img_tensor.float() / 255.0  # 归一化
    if img_tensor.ndimension() == 3: # 如果输入为单通道图片，增加batch维度
        img_tensor = img_tensor.unsqueeze(0)  # 加batch维度
    return img_tensor, img_letter, img0, (ratio, pad)

# -------------------------- 分类模型相关函数 --------------------------
def classify_transforms(size=224):
    """创建分类任务的图像预处理transforms"""
    scale_size = (size, size) if isinstance(size, int) else size
    tfl = [T.Resize(scale_size[0], interpolation=T.InterpolationMode.BILINEAR)]
    tfl += [T.CenterCrop(size), T.ToTensor(), T.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))]
    return T.Compose(tfl)

# -------------------------- 绘制检测结果：根据分类结果绘制不同颜色的框 --------------------------
def draw_detection_with_classification(img, xyxy, conf, landmarks, class_name, class_prob):
    """
    绘制检测结果：人脸框+5个关键点+置信度+分类结果
    参数:
        img: 图像
        xyxy: 人脸框坐标
        conf: 检测置信度
        landmarks: 关键点坐标
        class_name: 分类结果 ('Drowsy' 或 'Normal')
        class_prob: 分类概率
    """
    h, w = img.shape[:2]
    tl = round(0.002 * (h + w) / 2) + 1
    x1, y1, x2, y2 = map(int, xyxy)
    
    # 根据分类结果选择颜色：Drowsy=红色，Normal=绿色
    if class_name == 'Drowsy':
        box_color = (0, 0, 255)  # 红色 (BGR格式)
    else:  # Normal
        box_color = (0, 255, 0)  # 绿色 (BGR格式)
    
    # 绘制人脸框
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness=tl, lineType=cv2.LINE_AA)
    
    # 绘制关键点
    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    for i in range(5):
        px, py = int(landmarks[2*i]), int(landmarks[2*i+1])
        cv2.circle(img, (px, py), tl+1, clors[i], -1)
    
    # 绘制标签：检测置信度 + 分类结果
    tf = max(tl - 1, 1)
    label = f'{class_name} {class_prob:.2f}'
    cv2.putText(img, label, (x1, y1 - 2), 0, tl/3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    
    return img

# -------------------------- 核心推理函数：检测+分类 --------------------------
def face_detect_and_classify(detect_model, classify_model, img, device, classify_transforms, 
                              img_size=640, conf_thres=0.6, iou_thres=0.5):
    """
    人脸检测和分类
    参数:
        detect_model: 人脸检测模型
        classify_model: 分类模型
        img: 输入图像
        device: 设备
        classify_transforms: 分类模型的预处理transforms
        img_size: 检测模型输入尺寸
        conf_thres: 检测置信度阈值
        iou_thres: IOU阈值
    返回:
        绘制了检测和分类结果的图像
    """
    img_tensor, img_letter, img0, (ratio, pad) = preprocess_image(img, device, img_size)
    pred = detect_model(img_tensor)[0]
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)
    
    class_names = {0: 'Drowsy', 1: 'Normal'}  # 注意：根据2_test.py，1对应NonDrowsy，这里改为Normal
    
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img_letter.shape[:2], det[:, :4], img0.shape, (ratio, pad)).round()
            det[:, 5:15] = scale_coords_landmarks(img_letter.shape[:2], det[:, 5:15], img0.shape, (ratio, pad)).round()
            
            for j in range(det.size(0)):
                xyxy = det[j, :4].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # 提取人脸区域
                face_roi = img0[y1:y2, x1:x2]
                
                # 确保人脸区域有效
                if face_roi.size == 0:
                    continue
                
                # 将人脸区域转换为PIL图像并进行分类
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)
                
                # 分类预处理和推理
                with torch.no_grad():
                    input_tensor = classify_transforms(face_pil).unsqueeze(0).to(device)
                    probs, logits = classify_model(input_tensor)
                    probs = probs[0]  # 取第一个batch
                    
                    # 获取分类结果
                    top1_prob, top1_idx = torch.max(probs, 0)
                    class_name = class_names[top1_idx.item()]
                    class_prob = top1_prob.item()
                
                # 获取检测信息
                conf = det[j, 4].cpu().item()
                landmarks = det[j, 5:15].tolist()
                
                # 绘制结果（使用分类结果决定颜色）
                img0 = draw_detection_with_classification(img0, xyxy, conf, landmarks, class_name, class_prob)
    
    return img0

# -------------------------- 主程序 --------------------------
if __name__ == '__main__':
    # 核心配置
    DETECT_WEIGHTS_PATH = './weights/detect.pt'
    CLASSIFY_WEIGHTS_PATH = './weights/classify.pt'
    CONF_THRES = 0.6
    CLASSIFY_IMG_SIZE = 224
    DEBUG = True

    # 设备适配
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载人脸检测模型
    print("正在加载人脸检测模型...")
    detect_model = attempt_load(DETECT_WEIGHTS_PATH, map_location=device)
    print("人脸检测模型加载成功！")
    
    # 加载分类模型
    print("正在加载分类模型...")
    classify_checkpoint = torch.load(CLASSIFY_WEIGHTS_PATH, map_location=device, weights_only=False)
    classify_model = classify_checkpoint['model'].float().to(device).eval()
    classify_transforms = classify_transforms(size=CLASSIFY_IMG_SIZE)
    print("分类模型加载成功！")
    
    print("开始实时人脸检测和分类，按 'q' 键退出...")

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头！")
        sys.exit(1)

    # 帧数计算初始化
    fps_count = 0
    fps_start = time.time()
    fps = 0.0

    # 帧循环推理
    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取摄像头帧！")
            break
        
        # 帧数计算逻辑
        fps_count += 1
        current_time = time.time()
        if current_time - fps_start >= 1.0:  # 每1秒计算一次帧数
            fps = fps_count / (current_time - fps_start)
            fps_count = 0
            fps_start = current_time
        
        # 人脸检测和分类
        frame_result = face_detect_and_classify(
            detect_model, classify_model, frame, device, classify_transforms, 
            conf_thres=CONF_THRES
        )

        if DEBUG:
            print(f"frame_result: {frame_result.shape}")

        # 绘制帧数到画面左上角
        h, w = frame_result.shape[:2]
        tl = round(0.002 * (h + w) / 2) + 1
        tf = max(tl - 1, 1)
        cv2.putText(frame_result, f'FPS: {fps:.1f}', (10, 30), 0, tl/3, [0, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        # 显示结果
        cv2.imshow('Face Detection & Classification (Drowsy=Red, Normal=Green)', frame_result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("检测结束")

