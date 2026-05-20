import sys
from ultralytics import YOLO
from cvt_onnx import reorder_convert

ptname = sys.argv[1]

Yolo = YOLO(model=ptname).to('cpu')
Yolo.export(format='onnx', imgsz=[160,320] )

onnxName = ptname.replace('.pt', '.onnx')
reordered_onnxName = onnxName.replace('.onnx', '-reorder.onnx')
csv_name = onnxName.replace('.onnx', '.network.csv')
txt_name = onnxName.replace('.onnx', '.weights.txt')
reorder_convert(onnxName,
                "320x160.order.txt",
                reordered_onnxName,
                ["/model.27/Concat_2", "/model.27/Concat_1", "/model.27/Concat"],
                csv_name,
                txt_name)