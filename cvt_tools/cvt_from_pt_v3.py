import sys
from ultralytics import YOLO
from cvt_onnx import reorder_convert

ptname = sys.argv[1]

Yolo = YOLO(model=ptname).to('cpu')
Yolo.export(format='onnx', imgsz=[192, 192])

onnxName = ptname.replace('.pt', '.onnx')
reordered_onnxName = onnxName.replace('.onnx', '-reorder.onnx')
csv_name = onnxName.replace('.onnx', '.network.csv')
txt_name = onnxName.replace('.onnx', '.weights.txt')

# 关键：先不要 end_names 截断
reorder_convert(
    onnxName,
    "v3.order.txt",
    reordered_onnxName,
    [],
    csv_name,
    txt_name
)