import onnx
import copy
from OnnxModelExport import Exporter, Layer


class RamUsage:
	def __init__(self, name, size, start) -> None:
		self.name = name
		self.size = size
		if start >= 0:
			self.start = start
			self.end = start + 1

	name = ""
	size = 0
	start = 0
	end = 65535
	addr:int = 0

class LayerRamUsage:
	def __init__(self, layer:Layer) -> None:
		self.layer = layer
		self.total = 0
		self.usages = []
	layer:Layer
	usages:list[RamUsage] = []
	total:int = 0

def getTotal(usage:LayerRamUsage):
	return usage.total

def evaluateRamUsage(exporter:Exporter, program_size, kernel_size, image_size, result_size, end_layers:dict):
	usage0 = RamUsage("program", program_size, -1)
	usage1 = RamUsage("kernels", kernel_size, -1)
	#usage2 = RamUsage("images", image_size, -1)
	usage3 = RamUsage("result_box", result_size, -1)

	ram_usages = {str:RamUsage}
	ram_usages.clear()
	#ram_usages[usage0.name] = usage0
	#ram_usages[usage1.name] = usage1
	##ram_usages[usage2.name] = usage2
	#ram_usages[usage3.name] = usage3


	# 计算每层内存占用
	for layer in exporter.layers:
		out_size = layer.outputs[0].w*layer.outputs[0].h*layer.outputs[0].c
		for output in layer.outputs:
			usage = RamUsage(layer.name + "-" + output.format, out_size, layer.idx)
			usage.end = output.ref
			if usage.end == 0: 
				#输出完全没有被引用过,说明是最终输出层,输出需要常驻内存
				end = end_layers.get(layer.idx)
				if end is not None:
					usage.end = end
				else:
					usage.end = len(exporter.layers)
			
			ram_usages[usage.name] = usage

	layer_usages:list[LayerRamUsage] = []
	for layer in exporter.layers:
		layer_usage = LayerRamUsage(layer)
		out_size = layer.outputs[0].w*layer.outputs[0].h*layer.outputs[0].c*2

		covered = False
		for output in layer.outputs:
			## 满足以下条件可以考虑输出覆盖输入
			#if output.format == "any" or output.format == layer.inputs[0].format:
			#	for usage in ram_usages.values():
			#		if usage in layer_usage.usages:
			#			continue
			#		if usage.size >= out_size and usage.end == layer.idx and not covered:
			#			layer_usage.usages.append(RamUsage(usage.name + " output covered", 0, usage.start))
			#			covered = True
			#		elif layer.idx >= usage.start and usage.end >= layer.idx:
			#			layer_usage.usages.append(usage)
			#			layer_usage.total += usage.size
			#else:
			for usage in ram_usages.values():
				if usage in layer_usage.usages:
					continue
				if layer.idx >= usage.start and usage.end >= layer.idx:
					layer_usage.usages.append(usage)
					layer_usage.total += usage.size

		layer_usages.append(copy.copy(layer_usage))
		layer_usage = None
	# 排序, 方便评估最大内存占用
	#layer_usages.sort(key=getTotal)#, reverse=True)

	for layer_usage in layer_usages:
		layer = layer_usage.layer
		print("layer", layer.idx, "(", layer.name + ") used", layer_usage.total, "bytes(", layer_usage.total/1024.0, "kb)")

		for usage in layer_usage.usages:
			print("\t[", usage.name, int(usage.size), int(usage.start), "]")

mm = onnx.load("../UpixNetwork/model/HeadShoulder_320x160-reorder.onnx")
#for node in mm.graph.node:
#	if node.op_type == "Conv":
#		print(node.name)


exp = Exporter()
#exp.end_names = ["/model.27/Concat_2", "/model.27/Concat_1", "/model.27/Concat"]
exp.end_names = ["/model.15/Concat", "/model.15/Concat_1"]
exp.loadModel(mm)
#endLayers = {125:125, 120:125, 98:98, 93:98, 73:73, 68:73}
endLayers = {64:64, 59:64, 48:48, 43:48}
evaluateRamUsage(exp, 20*1024, 76240, 160*128, 32*6*4+16, endLayers)
