import copy
import cv2
import onnx
import numpy as np
import onnxruntime as ort
from onnx import helper

onnx_path = r"C:\u31-yolo\UpixNetwork\model\shabao_192x192-reorder.onnx"
image_path = r"C:\u31-yolo\UpixNetwork\data\shabao_test.jpg"

debug_onnx_path = r"C:\u31-yolo\UpixNetwork\model\shabao_192x192-debug-head.onnx"

extra_outputs = [
    "/model.15/Concat_output_0",
    "/model.15/Concat_1_output_0",
    "output0",
]

model = onnx.load(onnx_path)

# 添加中间输出
existing = {o.name for o in model.graph.output}
for name in extra_outputs:
    if name not in existing:
        model.graph.output.append(helper.ValueInfoProto(name=name))

onnx.save(model, debug_onnx_path)

img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
if img_bgr is None:
    raise RuntimeError(f"image not found: {image_path}")

img_bgr = cv2.resize(img_bgr, (192, 192))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

x = img_rgb.astype(np.float32) / 255.0
x = np.transpose(x, (2, 0, 1))[None, :, :, :]

sess = ort.InferenceSession(debug_onnx_path, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
output_names = [o.name for o in sess.get_outputs()]

outs = sess.run(output_names, {input_name: x})
out_map = dict(zip(output_names, outs))

print("available outputs:")
for k, v in out_map.items():
    print(k, v.shape, "min", v.min(), "max", v.max())

head24 = out_map["/model.15/Concat_output_0"]
head48 = out_map["/model.15/Concat_1_output_0"]
final = out_map["output0"]

print("\nhead24:", head24.shape)
print("head48:", head48.shape)
print("final:", final.shape)

# expected:
# head24: 1x43x24x24
# head48: 1x43x48x48
# final:  1x7x2880

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softmax(a, axis=0):
    a = a - np.max(a, axis=axis, keepdims=True)
    e = np.exp(a)
    return e / np.sum(e, axis=axis, keepdims=True)

def decode_head_nchw(head, stride):
    """
    head: 1 x 43 x H x W
    channels:
      0:3   cls logits
      3:43  dfl bbox, 4*10
    returns:
      boxes: 4 x N, cx cy w h
      scores: 3 x N
    """
    h = head.shape[2]
    w = head.shape[3]

    cls_logits = head[0, 0:3, :, :]
    dfl = head[0, 3:43, :, :]

    scores = sigmoid(cls_logits).reshape(3, -1)

    # dfl: 40 x H x W -> 4 x 10 x H x W
    dfl = dfl.reshape(4, 10, h, w)
    prob = softmax(dfl, axis=1)

    bins = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    dist = np.sum(prob * bins, axis=1)  # 4 x H x W

    # dist order: l, t, r, b
    l = dist[0]
    t = dist[1]
    r = dist[2]
    b = dist[3]

    grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    anchor_x = grid_x.astype(np.float32) + 0.5
    anchor_y = grid_y.astype(np.float32) + 0.5

    x1 = (anchor_x - l) * stride
    y1 = (anchor_y - t) * stride
    x2 = (anchor_x + r) * stride
    y2 = (anchor_y + b) * stride

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = x2 - x1
    bh = y2 - y1

    boxes = np.stack([cx, cy, bw, bh], axis=0).reshape(4, -1)
    return boxes, scores

boxes24, scores24 = decode_head_nchw(head24, 8)
boxes48, scores48 = decode_head_nchw(head48, 4)

ours = np.concatenate(
    [
        np.concatenate([boxes24, boxes48], axis=1),
        np.concatenate([scores24, scores48], axis=1),
    ],
    axis=0,
)[None, :, :]

print("\nours:", ours.shape)
print("final:", final.shape)

diff = np.abs(ours - final)
print("diff mean:", diff.mean())
print("diff max :", diff.max())

# 打印 top boxes 对比
def top_dets(p, name, conf=0.15):
    p = p[0]
    boxes = p[0:4, :]
    scores = p[4:7, :]

    print(f"\n===== {name} top dets =====")
    idxs = np.argsort(scores.max(axis=0))[::-1][:20]
    for idx in idxs:
        cls = int(np.argmax(scores[:, idx]))
        score = float(scores[cls, idx])
        if score < conf:
            continue
        cx, cy, bw, bh = boxes[:, idx]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        print(
            f"idx={idx:4d} cls={cls} score={score:.4f} "
            f"xyxy=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})"
        )

top_dets(final, "onnx final")
top_dets(ours, "manual decode")