import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

import cv2
import torch
# pip install torchvision==0.17.0 --force-reinstall --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install ultralytics
from PIL import Image
import torchvision.transforms as T 

# 配置
MODEL_PATH = "./weights/classify.pt"
IMG_SIZE = 224
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CAMERA = 0

def classify_transforms(size=224):
    """创建分类任务的图像预处理transforms"""
    scale_size = (size, size) if isinstance(size, int) else size
    tfl = [T.Resize(scale_size[0], interpolation=T.InterpolationMode.BILINEAR)]
    tfl += [T.CenterCrop(size), T.ToTensor(), T.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))]
    return T.Compose(tfl)

# 加载模型
print(f"Using device: {DEVICE}")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model = checkpoint['model'].float().to(DEVICE).eval()

class_names = {0: 'Drowsy', 1: 'NonDrowsy'}
model_transforms = classify_transforms(size=IMG_SIZE)

# 实时检测
cap = cv2.VideoCapture(CAMERA)
frame_count = 0

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 预处理
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = model_transforms(pil_img).unsqueeze(0).to(DEVICE)
        
        # 推理
        probs, logits = model(input_tensor)
        probs = probs[0]  # 取第一个batch
        
        # 获取结果
        top1_prob, top1_idx = torch.max(probs, 0)
        prob_drowsy, prob_nondrowsy = probs[0].item(), probs[1].item()
        class_name = class_names[top1_idx.item()]
        
        # 显示
        label1 = f"{class_name}: {top1_prob.item():.3f}"
        label2 = f"Drowsy: {prob_drowsy:.3f} | NonDrowsy: {prob_nondrowsy:.3f}"
        cv2.putText(frame, label1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, label2, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Real-time Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("检测结束")
