import onnx

onnx_path = r"C:\u31-yolo\UpixNetwork\model\shabao_192x192-reorder.onnx"

model = onnx.load(onnx_path)

print("graph outputs:")
for o in model.graph.output:
    print(" ", o.name)

print("\nlast 120 nodes:")
nodes = list(model.graph.node)
start = max(0, len(nodes) - 120)

for i in range(start, len(nodes)):
    node = nodes[i]
    print(
        f"{i:4d} | "
        f"{node.op_type:14s} | "
        f"name={node.name} | "
        f"inputs={list(node.input)} | "
        f"outputs={list(node.output)}"
    )