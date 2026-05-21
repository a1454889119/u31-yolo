import cv2
import numpy as np
import onnxruntime as ort

onnx_path = r"C:\u31-yolo\UpixNetwork\model\shabao_192x192-reorder.onnx"
image_path = r"C:\u31-yolo\UpixNetwork\data\shabao_test.jpg"
output_path = r"C:\u31-yolo\UpixNetwork\data\shabao_test_onnx_result.jpg"

class_names = {
    0: "red",
    1: "blue",
    2: "green",
}

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def nms_xyxy(dets, iou_thres=0.5):
    dets = sorted(dets, key=lambda x: x["score"], reverse=True)
    keep = []

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a["xyxy"]
        bx1, by1, bx2, by2 = b["xyxy"]

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - inter
        if union <= 0:
            return 0
        return inter / union

    for d in dets:
        ok = True
        for k in keep:
            if d["cls"] == k["cls"] and iou(d, k) > iou_thres:
                ok = False
                break
        if ok:
            keep.append(d)

    return keep

img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
if img_bgr is None:
    raise RuntimeError(f"image not found: {image_path}")

img_bgr = cv2.resize(img_bgr, (192, 192))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

x = img_rgb.astype(np.float32) / 255.0
x = np.transpose(x, (2, 0, 1))[None, :, :, :]

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
output_names = [o.name for o in sess.get_outputs()]

print("input:", input_name, x.shape)
print("outputs:")
for o in sess.get_outputs():
    print(" ", o.name, o.shape)

outs = sess.run(output_names, {input_name: x})

for i, out in enumerate(outs):
    print(f"out[{i}] name={output_names[i]} shape={out.shape} min={out.min()} max={out.max()}")

# 默认取第一个输出
p = outs[0][0]

# 期望 shape: 7 x 2880
if p.shape[0] != 7 and p.shape[1] == 7:
    p = p.T

print("decoded p shape:", p.shape)

boxes = p[0:4, :]
scores_raw = p[4:7, :]

print("scores raw min/max:", scores_raw.min(), scores_raw.max())

# 如果分数不在 0~1，做 sigmoid
if scores_raw.max() > 1.0 or scores_raw.min() < 0.0:
    scores = sigmoid(scores_raw)
    print("score mode: sigmoid")
else:
    scores = scores_raw
    print("score mode: raw")

conf_thres = 0.15
dets = []

for i in range(p.shape[1]):
    cls = int(np.argmax(scores[:, i]))
    score = float(scores[cls, i])

    if score < conf_thres:
        continue

    cx = float(boxes[0, i])
    cy = float(boxes[1, i])
    w = float(boxes[2, i])
    h = float(boxes[3, i])

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    x1 = max(0, min(191, x1))
    y1 = max(0, min(191, y1))
    x2 = max(0, min(191, x2))
    y2 = max(0, min(191, y2))

    if x2 <= x1 or y2 <= y1:
        continue

    dets.append({
        "cls": cls,
        "score": score,
        "xyxy": (x1, y1, x2, y2),
    })

dets = nms_xyxy(dets, 0.5)

print("\n===== ONNX result =====")
for d in dets:
    cls = d["cls"]
    score = d["score"]
    x1, y1, x2, y2 = d["xyxy"]

    print(
        f"cls={cls}({class_names.get(cls)}), "
        f"score={score:.4f}, "
        f"xyxy=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})"
    )

    color = (0, 255, 255)
    if cls == 0:
        color = (0, 0, 255)
    elif cls == 1:
        color = (255, 0, 0)
    elif cls == 2:
        color = (0, 255, 0)

    x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
    cv2.rectangle(img_bgr, (x1i, y1i), (x2i, y2i), color, 2)
    cv2.putText(
        img_bgr,
        f"{class_names.get(cls)} {score:.2f}",
        (x1i, max(15, y1i - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
    )

cv2.imwrite(output_path, img_bgr)
print("saved:", output_path)

cv2.imshow("onnx result", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()