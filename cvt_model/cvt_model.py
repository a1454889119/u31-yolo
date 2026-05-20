import onnx
from OnnxModelExport import Exporter
from OnnxReorder import reorder_onnx_nodes


def run416x320():
	mm = onnx.load("416x320.onnx")
	exp = Exporter()
	exp.end_names = ["Concat_257", "Concat_272", "Concat_287"]
	exp.loadModel(mm)
	exp.writeNetwork("416x320.network.csv")
	exp.writeWeights("416x320.weights.txt")

def run160x120():
	mm = onnx.load("../UpixNetwork/model/128x160-reorder.onnx")
	exp = Exporter()
	exp.end_names = ["/model.15/Concat", "/model.15/Concat_1"]
	exp.loadModel(mm)
	exp.writeNetwork("../UpixNetwork/model/128x160.network.csv")
	exp.writeWeights("../UpixNetwork/model/128x160.weights.txt")

def run320x160():
	mm = onnx.load("../UpixNetwork/model/Pool_320x160-reorder.onnx")
	#for node in mm.graph.node:
	#	print(node.name)

	exp = Exporter()
	exp.end_names = ["/model.27/Concat_2", "/model.27/Concat_1", "/model.27/Concat"]
	exp.loadModel(mm)
	exp.writeNetwork("../UpixNetwork/model/Pool_320x160.network.csv")
	exp.writeWeights("../UpixNetwork/model/Pool_320x160.weights.txt")	

def load_order(filename:str) -> list[str] :
	names:list[str] = []
	with open(filename) as file:
		while True:
			text = file.readline()
			if text == "":
				break
			names.append(text.replace("\n", ""))
	return names

def reorder_320x160(order_name:str):
	new_order = load_order(order_name)
	reorder_onnx_nodes("../UpixNetwork/model/Pool_320x160.onnx", "../UpixNetwork/model/Pool_320x160-reorder.onnx", new_order)

def reorder_160x128(order_name:str):
	new_order = load_order(order_name)
	reorder_onnx_nodes("../UpixNetwork/model/128x160.onnx", "../UpixNetwork/model/128x160-reorder.onnx", new_order)


def reorder_convert(model_name, order_name, reorder_name, end_names, output_network, output_weights):
	new_order = load_order(order_name)
	reorder_onnx_nodes(model_name, reorder_name, new_order)
	exp = Exporter()
	exp.end_names = end_names
	mm = onnx.load(reorder_name)
	exp.loadModel(mm)
	exp.writeNetwork(output_network)
	exp.writeWeights(output_weights)
	#exp.writeLayerNames("../UpixNetwork/model/Pool_320x160.layername.txt")



def print_layers(model_name):
	model = onnx.load(model_name)
	for node in model.graph.node:
		print(node.name)

#print_layers("../UpixNetwork/model/160x128.onnx")
#reorder_320x160("320x160.order.txt")
#run320x160()

#reorder_160x128("160x128.order.txt")
#run160x120()

#reorder_320x160("320x160_pool.order.txt")
#run320x160()

#print_layers("../UpixNetwork/model/Pool_320x160.onnx")

#reorder_convert("../UpixNetwork/model/320x160.onnx",
#		"320x160.order.txt",
#		"../UpixNetwork/model/320x160-reorder.onnx",
#		["/model.27/Concat_2", "/model.27/Concat_1", "/model.27/Concat"],
#		"../UpixNetwork/model/320x160.network.csv",
#		"../UpixNetwork/model/320x160.weights.txt")

# pool model
#reorder_convert("../UpixNetwork/model/Pool_320x160.onnx",
#		"320x160_pool.order.txt",
#		"../UpixNetwork/model/Pool_320x160-reorder.onnx",
#		["/model.30/Concat_2", "/model.30/Concat_1", "/model.30/Concat"],
#		"../UpixNetwork/model/Pool_320x160.network.csv",
#		"../UpixNetwork/model/Pool_320x160.weights.txt")

# facepalm model
#reorder_convert("../UpixNetwork/model/320x160_facepalm.onnx",
#		"320x160.order.txt",
#		"../UpixNetwork/model/320x160_facepalm-reorder.onnx",
#		["/model.27/Concat_2", "/model.27/Concat_1", "/model.27/Concat"],
#		"../UpixNetwork/model/320x160_facepalm.network.csv",
#		"../UpixNetwork/model/320x160_facepalm.weights.txt")

#reorder_convert("../UpixNetwork/model/facepalm_320x160_3types.onnx",
#		"320x160.order.txt",
#		"../UpixNetwork/model/facepalm_320x160_3types-reorder.onnx",
#		["/model.27/Concat_2", "/model.27/Concat_1", "/model.27/Concat"],
#		"../UpixNetwork/model/facepalm_320x160_3types.network.csv",
#		"../UpixNetwork/model/facepalm_320x160_3types.weights.txt")

#reorder_convert("../UpixNetwork/model/HeadShoulder_160x128.onnx",
#		"160x128.order.txt",
#		"../UpixNetwork/model/HeadShoulder_160x128-reorder.onnx",
#		["/model.15/Concat", "/model.15/Concat_1"],
#		"../UpixNetwork/model/HeadShoulder_160x128.network.csv",
#		"../UpixNetwork/model/HeadShoulder_160x128.weights.txt")

#reorder_convert("../UpixNetwork/model/HeadShoulder_320x160.onnx",
#		"160x128.order.txt",
#		"../UpixNetwork/model/HeadShoulder_320x160-reorder.onnx",
#		["/model.15/Concat", "/model.15/Concat_1"],
#		"../UpixNetwork/model/HeadShoulder_320x160.network.csv",
#		"../UpixNetwork/model/HeadShoulder_320x160.weights.txt")

reorder_convert("../UpixNetwork/model/Pool_640x480_4class_Gray.onnx",
		"320x160.order.txt",
		"../UpixNetwork/model/Pool_640x480_4class_Gray-reorder.onnx",
		["/model.27/Concat_2", "/model.27/Concat_1", "/model.27/Concat"],
		"../UpixNetwork/model/Pool_640x480_4class_Gray.network.csv",
		"../UpixNetwork/model/Pool_640x480_4class_Gray.weights.txt")