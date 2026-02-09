#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from rknn.api import RKNN
from PIL import Image
import numpy as np
import cv2
import time
from typing import Final, Tuple

# -------------------------- 配置参数 --------------------------
DETECT_ONNX_PATH: Final[str] = "./weights/face.onnx"
CLASSIFY_ONNX_PATH: Final[str] = "./weights/face_classification.onnx"
PLATFORM = 'rk3588'
DETECT_IMG_SIZE: Final[int] = 320
CLASSIFY_INPUT_SIZE: Final[Tuple[int, int]] = (224, 224)
DETECT_INPUT_SIZE_LIST = [[3, DETECT_IMG_SIZE, DETECT_IMG_SIZE]]
CLASSIFY_INPUT_SIZE_LIST = [[3, *CLASSIFY_INPUT_SIZE]]
CONF_THRES = 0.70
IOU_THRES = 0.5
DEVICE = "/dev/video11"  # 摄像头设备号（根据实际修改）
DEBUG = False
class_names = {0: 'Drowsy', 1: 'Normal'}

# 全局RKNN模型实例
detect_rknn_model = None
classify_rknn_model = None

# -------------------------- 核心工具函数 --------------------------
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False):
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

def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def box_iou(box1, box2):
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])
    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    inter = (np.min(box1[:, None, 2:], box2[:, 2:]) - np.max(box1[:, None, :2], box2[:, :2])).clip(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)

def non_max_suppression_face(prediction, conf_thres=0.25, iou_thres=0.45):
    nc = prediction.shape[2] - 15
    xc = prediction[..., 4] > conf_thres
    output = [np.zeros((0, 16), dtype=np.float32)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        x[:, 15:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        conf = np.max(x[:, 15:], axis=1, keepdims=True)
        j = np.argmax(x[:, 15:], axis=1, keepdims=True)
        x = np.concatenate((box, conf, x[:, 5:15], j.astype(np.float32)), axis=1)[conf.ravel() > conf_thres]
        n = x.shape[0]
        if not n:
            continue
        scores = x[:, 4]
        boxes = x[:, :4]
        indices = cv2.dnn.NMSBoxes(boxes[:, :4].tolist(), scores.tolist(), conf_thres, iou_thres)
        if indices.size > 0:
            indices = indices.ravel()
            output[xi] = x[indices]
    return output

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
    coords[:, 0] = np.clip(coords[:, 0], 0, img0_shape[1])
    coords[:, 1] = np.clip(coords[:, 1], 0, img0_shape[0])
    coords[:, 2] = np.clip(coords[:, 2], 0, img0_shape[1])
    coords[:, 3] = np.clip(coords[:, 3], 0, img0_shape[0])
    return coords

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
    coords[:, 0] = np.clip(coords[:, 0], 0, img0_shape[1])
    coords[:, 1] = np.clip(coords[:, 1], 0, img0_shape[0])
    coords[:, 2] = np.clip(coords[:, 2], 0, img0_shape[1])
    coords[:, 3] = np.clip(coords[:, 3], 0, img0_shape[0])
    coords[:, 4] = np.clip(coords[:, 4], 0, img0_shape[1])
    coords[:, 5] = np.clip(coords[:, 5], 0, img0_shape[0])
    coords[:, 6] = np.clip(coords[:, 6], 0, img0_shape[1])
    coords[:, 7] = np.clip(coords[:, 7], 0, img0_shape[0])
    coords[:, 8] = np.clip(coords[:, 8], 0, img0_shape[1])
    coords[:, 9] = np.clip(coords[:, 9], 0, img0_shape[0])
    return coords

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
    
    # 绘制标签：分类结果 + 分类概率
    tf = max(tl - 1, 1)
    label = f'{class_name} {class_prob:.2f}'
    cv2.putText(img, label, (x1, y1 - 2), 0, tl/3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    
    return img

# -------------------------- RKNN模型相关函数 --------------------------
def generate_rknn(onnx_path, input_size_list, target_platform="rk3588"):
    rknn = RKNN(verbose=False)
    if rknn.config(mean_values=[[0,0,0]], std_values=[[1,1,1]], target_platform=target_platform) != 0:
        print(f"[ERROR] 配置RKNN失败")
        rknn.release()
        exit(1)
    if rknn.load_onnx(model=onnx_path, input_size_list=input_size_list) != 0:
        print(f"[ERROR] 加载ONNX模型{onnx_path}失败")
        rknn.release()
        exit(1)
    if rknn.build(do_quantization=False, dataset=None) != 0:
        print(f"[ERROR] 构建RKNN模型失败")
        rknn.release()
        exit(1)
    rknn_path = onnx_path.replace(".onnx", ".rknn")
    if rknn.export_rknn(rknn_path) != 0:
        print(f"[ERROR] 导出RKNN模型{rknn_path}失败")
        rknn.release()
        exit(1)
    print(f"[SUCCESS] RKNN模型生成并导出成功：{rknn_path}")
    rknn.release()

def load_rknn_model(onnx_path, input_size_list, target_platform="rk3588"):
    rknn_path = onnx_path.replace(".onnx", ".rknn")
    rknn = RKNN(verbose=False)
    
    # 优先加载本地已存在的rknn模型
    if os.path.exists(rknn_path):
        print(f"[INFO] 检测到本地RKNN模型，开始加载：{rknn_path}")
        if rknn.load_rknn(rknn_path) != 0:
            print(f"[ERROR] 加载本地RKNN模型{rknn_path}失败")
            rknn.release()
            return None
    # 本地无rknn模型，从ONNX生成后再加载
    else:
        print(f"[INFO] 未检测到本地RKNN模型，开始从ONNX生成...")
        generate_rknn(onnx_path, input_size_list, target_platform)
        if rknn.load_rknn(rknn_path) != 0:
            print(f"[ERROR] 加载生成的RKNN模型{rknn_path}失败")
            rknn.release()
            return None
    
    # 初始化RKNN运行时
    print(f"[INFO] 初始化RKNN运行时...")
    if rknn.init_runtime(target=target_platform, core_mask=RKNN.NPU_CORE_ALL) != 0:
        print(f"[ERROR] 初始化RKNN运行时失败")
        rknn.release()
        return None
    print(f"[SUCCESS] RKNN模型加载+运行时初始化完成！")
    return rknn

# -------------------------- 图像预处理函数 --------------------------
def preprocess_detect_image(img_frame, img_size=320):
    """人脸检测模型的图像预处理"""
    img0 = img_frame.copy()
    img = img0.copy()
    img_letter, ratio, pad = letterbox(img, new_shape=(img_size, img_size))
    img_rgb = img_letter[:, :, ::-1]  # BGR->RGB
    img_trans = img_rgb.transpose(2, 0, 1)  # HWC->CHW
    img_np = np.ascontiguousarray(img_trans, dtype=np.float32) / 255.0  # 归一化+连续数组
    img_np = np.expand_dims(img_np, axis=0)  # 增加batch维度（NCHW）
    return img_np, img_letter, img0, (ratio, pad)

def preprocess_classify_image(face_roi):
    """
    分类模型的图像预处理（按照提取流程：先BGR转RGB，再Resize + CenterCrop）
    参数:
        face_roi: BGR格式的numpy数组（从img0[y1:y2, x1:x2]提取）
    返回:
        numpy数组，形状为 (1, 224, 224, 3)，dtype=float32，RGB格式，NHWC
    """
    # 确保人脸区域有效
    if face_roi.size == 0:
        # 如果区域无效，返回一个黑色图像
        img = np.zeros((*CLASSIFY_INPUT_SIZE, 3), dtype=np.uint8)
    else:
        # 按照你提供的流程：先BGR转RGB
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)  # BGR->RGB
        
        h, w = face_rgb.shape[:2]
        target_size = CLASSIFY_INPUT_SIZE[0]  # 224
        
        # 先resize，保持宽高比，短边缩放到target_size
        if h < w:
            new_h = target_size
            new_w = int(w * target_size / h)
        else:
            new_w = target_size
            new_h = int(h * target_size / w)
        
        img = cv2.resize(face_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # CenterCrop到224x224
        h, w = img.shape[:2]
        start_h = (h - target_size) // 2
        start_w = (w - target_size) // 2
        img = img[start_h:start_h + target_size, start_w:start_w + target_size]
    
    # 归一化：(img / 255.0 - mean) / std
    img = (img / 255.0 - np.array([0.0, 0.0, 0.0])) / np.array([1.0, 1.0, 1.0]) # mean=[0,0,0], std=[1,1,1] 实际上就是 img/255.0
    
    # 确保数据类型和形状正确
    img = np.ascontiguousarray(img, dtype=np.float32)
    return np.expand_dims(img, axis=0)

# -------------------------- 分类推理函数 --------------------------
def classify_face(classify_rknn, face_roi):
    """
    对检测到的人脸区域进行分类
    返回: (class_name, class_prob)
    """
    if classify_rknn is None:
        return 'Normal', 0.5
    
    # 预处理
    input_tensor = preprocess_classify_image(face_roi)
    
    # RKNN推理
    outputs = classify_rknn.inference(inputs=[input_tensor], data_format='nhwc')
    probs = outputs[0][0]

    # 提取概率值
    prob_nondrowsy = float(probs[1]) # probs[1] 是不瞌睡的概率
    prob_drowsy = float(probs[0]) # probs[0] 是不瞌睡的概率
    
    if DEBUG:
        print(f"prob_nondrowsy: {prob_nondrowsy}, prob_drowsy: {prob_drowsy}")
    
    # 判断是否为瞌睡
    is_drowsy_flag = prob_drowsy > prob_nondrowsy
    class_name = class_names[0] if is_drowsy_flag else class_names[1]  # 0: Drowsy, 1: Normal
    class_prob = prob_drowsy if is_drowsy_flag else prob_nondrowsy
    
    return class_name, class_prob

# -------------------------- 核心推理函数：检测+分类 --------------------------
def face_detect_and_classify_rknn(
    detect_rknn, 
    classify_rknn,
    img_frame, 
    img_size=320, 
    conf_thres=0.70, 
    iou_thres=0.5
):
    """
    人脸检测和分类（RKNN版本）
    参数:
        detect_rknn: 人脸检测RKNN模型
        classify_rknn: 分类RKNN模型
        img_frame: 输入图像帧
        img_size: 检测模型输入尺寸
        conf_thres: 检测置信度阈值
        iou_thres: IOU阈值
    返回:
        绘制了检测和分类结果的图像
    """
    # 图像预处理
    img_input, img_letter, img0, (ratio, pad) = preprocess_detect_image(img_frame, img_size)
    
    # RKNN推理（人脸检测）
    outputs = detect_rknn.inference(inputs=[img_input], data_format='nchw')
    pred = outputs[0]
    
    # NMS非极大值抑制
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)
    
    # 后处理+分类+绘制结果
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img_letter.shape[:2], det[:, :4], img0.shape, (ratio, pad)).round()
            det[:, 5:15] = scale_coords_landmarks(img_letter.shape[:2], det[:, 5:15], img0.shape, (ratio, pad)).round()
            
            for j in range(det.shape[0]):
                xyxy = det[j, :4].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # 提取人脸区域
                face_roi = img0[y1:y2, x1:x2]
                
                # 确保人脸区域有效
                if face_roi.size == 0:
                    continue
                
                # 对检测到的人脸进行分类
                class_name, class_prob = classify_face(classify_rknn, face_roi)
                
                # 获取检测信息
                conf = det[j, 4].item()
                landmarks = det[j, 5:15].tolist()
                
                # 绘制结果（使用分类结果决定颜色）
                img0 = draw_detection_with_classification(img0, xyxy, conf, landmarks, class_name, class_prob)
    
    return img0

# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    print(f"[INFO] 开始加载RKNN模型...")
    
    # 加载人脸检测RKNN模型
    print(f"[INFO] 正在加载人脸检测模型...")
    detect_rknn_model = load_rknn_model(DETECT_ONNX_PATH, DETECT_INPUT_SIZE_LIST, PLATFORM)
    if detect_rknn_model is None:
        print(f"[ERROR] 人脸检测模型加载失败，退出程序")
        exit(1)
    
    # 加载分类RKNN模型
    print(f"[INFO] 正在加载分类模型...")
    classify_rknn_model = load_rknn_model(CLASSIFY_ONNX_PATH, CLASSIFY_INPUT_SIZE_LIST, PLATFORM)
    if classify_rknn_model is None:
        print(f"[ERROR] 分类模型加载失败，退出程序")
        detect_rknn_model.release()
        exit(1)
    
    print(f"[SUCCESS] 所有模型加载完成！")
    
    # 初始化摄像头
    print(f"[INFO] 打开摄像头设备 {DEVICE}...")
    cap = cv2.VideoCapture(DEVICE)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头设备 {DEVICE}")
        detect_rknn_model.release()
        classify_rknn_model.release()
        exit(1)
    
    # 配置摄像头参数
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"[SUCCESS] 摄像头打开并配置成功！")
    
    # 实时摄像头检测循环（按q退出）
    print(f"[INFO] 开始实时检测...")
    
    fps_count = 0
    fps_start = time.time()
    fps = 0.0
    
    try:
        while True:
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                print(f"[WARN] 读取摄像头帧失败，退出循环")
                break
            
            # 人脸检测和分类推理
            det_frame = face_detect_and_classify_rknn(
                detect_rknn_model, 
                classify_rknn_model,
                frame, 
                img_size=DETECT_IMG_SIZE, 
                conf_thres=CONF_THRES, 
                iou_thres=IOU_THRES
            )
            
            # 计算FPS
            fps_count += 1
            current_time = time.time()
            if current_time - fps_start >= 1.0:  # 每1秒计算一次帧数
                fps = fps_count / (current_time - fps_start)
                fps_count = 0
                fps_start = current_time
            
            # 绘制FPS
            h, w = det_frame.shape[:2]
            tl = round(0.002 * (h + w) / 2) + 1
            tf = max(tl - 1, 1)
            cv2.putText(det_frame, f'FPS: {fps:.1f}', (10, 30), 0, tl/3, [0, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
            
            # 显示检测结果
            cv2.imshow('Drowsiness Detection System (demo)', det_frame)
            
            # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"[INFO] 用户按下 'q'，退出检测")
                break
    finally:
        print(f"[INFO] 释放所有资源...")
        cap.release()
        cv2.destroyAllWindows()
        if detect_rknn_model:
            detect_rknn_model.release()
        if classify_rknn_model:
            classify_rknn_model.release()
        print(f"[SUCCESS] 所有资源释放完成！")

