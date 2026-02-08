#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

import cv2
import numpy as np
from rknn.api import RKNN
from typing import Final, Tuple, Union

CLASSIFICATION_MODEL_PATH: Final[str] = "./weights/face_classification.onnx"
PLATFORM = 'rk3588'
INPUT_SIZE: Final[Tuple[int, int]] = (224, 224)
INPUT_SIZE_LIST = [[3, *INPUT_SIZE]]
class_names = {0: 'Drowsy', 1: 'NonDrowsy'}

# 全局RKNN模型实例
rknn_model = None

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

def init_model():
    global rknn_model
    rknn_path = CLASSIFICATION_MODEL_PATH.replace(".onnx", ".rknn")
    rknn_model = RKNN(verbose=False)
    
    # 优先加载本地RKNN模型
    if os.path.exists(rknn_path):
        print(f"[INFO] 检测到本地RKNN模型，开始加载：{rknn_path}")
        if rknn_model.load_rknn(rknn_path) != 0:
            print(f"[ERROR] 加载本地RKNN模型{rknn_path}失败")
            rknn_model.release()
            return False
    # 本地无RKNN模型，从ONNX生成后再加载
    else:
        print(f"[INFO] 未检测到本地RKNN模型，开始从ONNX生成...")
        generate_rknn(CLASSIFICATION_MODEL_PATH, INPUT_SIZE_LIST, PLATFORM)
        if rknn_model.load_rknn(rknn_path) != 0:
            print(f"[ERROR] 加载生成的RKNN模型{rknn_path}失败")
            rknn_model.release()
            return False
    
    # 初始化运行时
    print(f"[INFO] 初始化RKNN运行时...")
    if rknn_model.init_runtime(target=PLATFORM) != 0:
        print(f"[ERROR] 初始化RKNN运行时失败")
        rknn_model.release()
        return False
    print(f"[SUCCESS] RKNN模型加载+运行时初始化完成！")
    return True

def preprocess_image(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {img_path}")
    
    img = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img / 255.0 - np.array([0.0, 0.0, 0.0])) / np.array([1.0, 1.0, 1.0])
    return np.expand_dims(img, axis=0).astype(np.float32)

def is_drowsy(img_path: str) -> tuple:
    if rknn_model is None:
        raise RuntimeError("模型未初始化，请先调用init_model()")
    
    input_tensor = preprocess_image(img_path)
    outputs = rknn_model.inference(inputs=[input_tensor], data_format='nhwc')
    probs = outputs[0][0]
    
    # 核心修复：根据你的输出调整概率映射关系
    # 从你的输出看，probs[0]是非瞌睡概率，probs[1]是瞌睡概率
    prob_nondrowsy = float(probs[0])
    prob_drowsy = float(probs[1])
    is_drowsy_flag = prob_drowsy > prob_nondrowsy
    class_name = class_names[0] if is_drowsy_flag else class_names[1]
    
    return is_drowsy_flag, class_name, prob_drowsy, prob_nondrowsy

if __name__ == "__main__":
    if not init_model():
        exit(1)
    
    test_images = [
        './test_images/drowsy.png',
        './test_images/nodrowsy.png'
    ]
    
    for img_path in test_images:
        try:
            flag, name, d_prob, nd_prob = is_drowsy(img_path)
            print(f"\n图片: {img_path}")
            print(f"是否瞌睡: {'是' if flag else '否'}")
            print(f"分类结果: {name}")
            print(f"瞌睡概率: {d_prob:.3f} | 非瞌睡概率: {nd_prob:.3f}")
        except Exception as e:
            print(f"处理图片 {img_path} 出错: {e}")
    
    if rknn_model:
        rknn_model.release()
        print("\n[INFO] 释放RKNN资源...")
    print("[SUCCESS] 检测完成，资源已释放！")