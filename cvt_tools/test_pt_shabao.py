import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path

model_path = r"C:\u31-yolo\UpixNetwork\model\shabao_192x192.pt"
image_path = r"C:\u31-yolo\UpixNetwork\data\shabao_test.jpg"
output_path = r"C:\u31-yolo\UpixNetwork\data\shabao_test_pt_raw_result.jpg"

class_names = {
    0: "red",
    1: "blue",
    2: "green",
}

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def xywh_to_xyxy(x, y, w, h):
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter
    if union <= 0:
        return 0.0

    return inter / union

def nms(dets, iou_thres=0.5):
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    keep = []

    while dets:
        best = dets.pop(0)
        keep.append(best)

        remain = []
        for d in dets:
            if d["cls"] != best["cls"]:
                remain.append(d)
                continue

            if iou_xyxy(d["xyxy"], best["xyxy"]) <= iou_thres:
                remain.append(d)

        dets = remain

    return keep

# 1. 加载模型
yolo = YOLO(model_path)
model = yolo.model
model.eval()
model.cpu()

# 2. 读图，强制 RGB 3 通道，192x192
img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
if img_bgr is None:
    raise RuntimeError(f"image not found: {image_path}")

img_bgr = cv2.resize(img_bgr, (192, 192))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 3. 手动构造 BCHW Tensor: 1x3x192x192
x = img_rgb.astype(np.float32) / 255.0
x = np.transpose(x, (2, 0, 1))          # HWC -> CHW
x = np.expand_dims(x, axis=0)           # CHW -> BCHW
x = torch.from_numpy(x).float()

print("input tensor shape:", tuple(x.shape))

# 4. 前向
with torch.no_grad():
    pred = model(x)

# 有些模型返回 tuple/list
if isinstance(pred, (tuple, list)):
    pred0 = pred[0]
else:
    pred0 = pred

print("raw output type:", type(pred))
print("raw output shape:", tuple(pred0.shape))

# 你的导出日志显示 output shape 是 (1, 7, 2880)
# 7 = 4 bbox + 3 class
p = pred0[0].detach().cpu().numpy()     # shape: 7 x N

print("p shape:", p.shape)

# 如果是 N x 7，就转成 7 x N
if p.shape[0] != 7 and p.shape[1] == 7:
    p = p.T

boxes = p[0:4, :]       # x, y, w, h
scores_raw = p[4:7, :]  # 3 classes

# 注意：如果输出已经是 0~1 概率，不要 sigmoid。
# 这里先判断一下范围。
print("score raw min/max:", scores_raw.min(), scores_raw.max())

if scores_raw.max() > 1.0 or scores_raw.min() < 0.0:
    scores = sigmoid(scores_raw)
    print("using sigmoid for class scores")
else:
    scores = scores_raw
    print("using raw class scores")

conf_thres = 0.15

dets = []

for i in range(p.shape[1]):
    cls_id = int(np.argmax(scores[:, i]))
    score = float(scores[cls_id, i])

    if score < conf_thres:
        continue

    x_center = float(boxes[0, i])
    y_center = float(boxes[1, i])
    w = float(boxes[2, i])
    h = float(boxes[3, i])

    x1, y1, x2, y2 = xywh_to_xyxy(x_center, y_center, w, h)

    # clamp
    x1 = max(0.0, min(191.0, x1))
    y1 = max(0.0, min(191.0, y1))
    x2 = max(0.0, min(191.0, x2))
    y2 = max(0.0, min(191.0, y2))

    if x2 <= x1 or y2 <= y1:
        continue

    dets.append({
        "cls": cls_id,
        "score": score,
        "xyxy": (x1, y1, x2, y2),
    })

dets = nms(dets, iou_thres=0.5)

print("\n===== PT raw detect result =====")
print("num dets:", len(dets))

for d in dets:
    cls_id = d["cls"]
    score = d["score"]
    x1, y1, x2, y2 = d["xyxy"]

    print(
        f"cls={cls_id}({class_names.get(cls_id, 'unknown')}), "
        f"score={score:.4f}, "
        f"xyxy=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
    )

    if cls_id == 0:
        color = (0, 0, 255)
    elif cls_id == 1:
        color = (255, 0, 0)
    elif cls_id == 2:
        color = (0, 255, 0)
    else:
        color = (0, 255, 255)

    x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
    cv2.rectangle(img_bgr, (x1i, y1i), (x2i, y2i), color, 2)

    label = f"{class_names.get(cls_id, cls_id)} {score:.2f}"
    cv2.putText(
        img_bgr,
        label,
        (x1i, max(15, y1i - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(output_path, img_bgr)

print(f"\nsaved result to: {output_path}")

cv2.imshow("pt raw detect result", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()