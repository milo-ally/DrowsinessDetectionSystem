#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from rknn.api import RKNN
from PIL import Image
import numpy as np
import cv2

# -------------------------- 核心工具函数 --------------------------
def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

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

def draw_detection(img, xyxy, conf, landmarks):
    h, w = img.shape[:2]
    tl = round(0.002 * (h + w) / 2) + 1
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)
    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    for i in range(5):
        px, py = int(landmarks[2*i]), int(landmarks[2*i+1])
        cv2.circle(img, (px, py), tl+1, clors[i], -1)
    tf = max(tl - 1, 1)
    cv2.putText(img, f'{conf:.2f}', (x1, y1 - 2), 0, tl/3, [255,255,255], thickness=tf, lineType=cv2.LINE_AA)
    return img

# rknn preparation
def generate_rknn(onnx_path, input_size_list, target_platform="rk3588"):
    rknn = RKNN(verbose=False)

    # 配置
    if rknn.config(
        mean_values=[[0,0,0]], 
        std_values=[[1,1,1]], 
        target_platform=target_platform
        ) != 0:
        print(f"[ERROR] 配置RKNN失败")
        rknn.release()
        exit(1)

    # 加载ONNX
    if rknn.load_onnx(
        model=onnx_path, 
        input_size_list=input_size_list
        ) != 0:
        print(f"[ERROR] 加载ONNX模型{onnx_path}失败")
        rknn.release()
        exit(1)

    # 构建
    if rknn.build(
        do_quantization=False, 
        dataset=None) != 0:
        print(f"[ERROR] 构建RKNN模型失败")
        rknn.release()
        exit(1)

    # 导出rknn文件
    rknn_path = onnx_path.replace(".onnx", ".rknn")
    if rknn.export_rknn(rknn_path) != 0:
        print(f"[ERROR] 导出RKNN模型{rknn_path}失败")
        rknn.release()
        exit(1)
    print(f"[SUCCESS] RKNN模型生成并导出成功：{rknn_path}")
    rknn.release()

# 加载RKNN模型或者生成rknn模型之后加载
def load_rknn_model(onnx_path, input_size_list, target_platform="rk3588"):

    rknn_path = onnx_path.replace(".onnx", ".rknn")
    rknn = RKNN(verbose=False)

    # 优先加载本地rknn模型
    if os.path.exists(rknn_path):
        print(f"[INFO] 检测到本地RKNN模型，开始加载：{rknn_path}")
        if rknn.load_rknn(rknn_path) != 0:
            print(f"[ERROR] 加载本地RKNN模型{rknn_path}失败")
            rknn.release()
            exit(1)

    # 本地无rknn模型，从ONNX生成后再加载
    else:
        print(f"[INFO] 未检测到本地RKNN模型，开始从ONNX生成...")
        generate_rknn(onnx_path, input_size_list, target_platform)
        if rknn.load_rknn(rknn_path) != 0:
            print(f"[ERROR] 加载生成的RKNN模型{rknn_path}失败")
            rknn.release()
            exit(1)

    # 初始化运行时
    print(f"[INFO] 初始化RKNN运行时...")
    if rknn.init_runtime(target=target_platform) != 0:
        print(f"[ERROR] 初始化RKNN运行时失败")
        rknn.release()
        exit(1)
    print(f"[SUCCESS] RKNN模型加载+运行时初始化完成！")
    return rknn

# 图像预处理
def preprocess_image(img_path, img_size=640):
    img0 = cv2.imread(img_path)
    assert img0 is not None, f"[ERROR] 无法加载图像：{img_path}"
    img = img0.copy()
    img_letter, ratio, pad = letterbox(img, new_shape=(img_size, img_size))
    img_rgb = img_letter[:, :, ::-1]
    img_trans = img_rgb.transpose(2, 0, 1)
    img_np = np.ascontiguousarray(img_trans, dtype=np.float32) / 255.0
    img_np = np.expand_dims(img_np, axis=0)
    return img_np, img_letter, img0, (ratio, pad)

# 推理
def face_detect_rknn(
        rknn,
        img_path,
        img_size=640,
        conf_thres=0.6,
        iou_thres=0.5,
        save_results=False
    ):

    # 图像预处理
    img_input, img_letter, img0, (ratio, pad) = preprocess_image(img_path, img_size)

    # RKNN推理
    outputs = rknn.inference(inputs=[img_input], data_format='nchw')
    pred = outputs[0]

    # NMS非极大值抑制
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    # 后处理+绘制结果
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img_letter.shape[:2], det[:, :4], img0.shape, (ratio, pad)).round()
            det[:, 5:15] = scale_coords_landmarks(img_letter.shape[:2], det[:, 5:15], img0.shape, (ratio, pad)).round()
            for j in range(det.shape[0]):
                xyxy = det[j, :4].tolist()

                x1, y1, x2, y2 = map(int, xyxy)
                face_roi = img0[y1:y2, x1:x2]
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB) # BGR->RGB
                face_image = Image.fromarray(face_rgb) # numpy数组->PIL图像（这里扣出了人脸区域）
                print(f"face_image's width: {face_image.size[0]}, height: {face_image.size[1]}") # face_image ->(W, H)
                
                conf = det[j, 4].item()
                landmarks = det[j, 5:15].tolist()
                img0 = draw_detection(img0, xyxy, conf, landmarks)

    # 保存结果
    if save_results:
        os.makedirs('face_detection_results', exist_ok=True)
        save_path = f'face_detection_results/rknn_detect_result.jpg'
        cv2.imwrite(save_path, img0)
        print(f"[SUCCESS] 检测结果已保存：{save_path}")
    return img0


if __name__ == "__main__":

    ONNX_PATH = "./weights/face.onnx"   # ONNX模型路径
    PLATFORM = 'rk3588'                 # 运行平台
    CONF_THRES = 0.6                    # 检测置信度阈值
    IOU_THRES = 0.5                     # NMS的IOU阈值
    IMG_SIZE = 320                      # 输入图像尺寸
    INPUT_SIZE_LIST = [[3, IMG_SIZE, IMG_SIZE]] # RKNN输入尺寸
    IMG_PATH = './test_images/manyfaces.jpg'    # 测试图像路径

    # 核心：加载RKNN模型（自动判断+生成）
    rknn_model = load_rknn_model(ONNX_PATH, INPUT_SIZE_LIST, PLATFORM)

    # 开始人脸检测
    print(f"[INFO] 开始RKNN人脸检测...")
    try:
        face_detect_rknn(
            rknn_model, 
            IMG_PATH, 
            img_size=IMG_SIZE,
            conf_thres=CONF_THRES, 
            iou_thres=IOU_THRES, 
            save_results=True
        )
    finally:
        print(f"[INFO] 释放RKNN资源...")
        rknn_model.release()
        print(f"[SUCCESS] 检测完成，资源已释放！")