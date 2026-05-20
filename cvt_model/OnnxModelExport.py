import onnx
import copy
import numpy

from LayerShape import Layer, WeightShap, IOShape

class Exporter:
	biases = {}
	weights = {}
	ioshapes = {}
	layer_out_dict = {}
	layers:list[Layer] = []
	temp_layers:list[Layer] = []
	end_names:list[str] = []
	constant_values = {}
	gathers = {}
	initializers = {}

	def loadModel(self, model:onnx.ModelProto):
		self.loadInput(model)
		self.loadInitializer(model)
		self.loadConstant(model)
		self.loadNodes(model)
		self.checkIOFormat()

	def loadInitializer(self, model:onnx.ModelProto):
		for initializer in model.graph.initializer:
			#print(initializer.name, initializer.dims)
			if initializer.name.endswith("bias"):
				self.biases[initializer.name] = initializer.dims[0]
				#print(initializer.name, initializer.dims)
			elif initializer.name.endswith("weight"):
				s = WeightShap()
				s.name = initializer.name
				s.inc = initializer.dims[1]
				s.outc = initializer.dims[0]
				s.kx = initializer.dims[2]
				s.ky = initializer.dims[3]
				self.weights[initializer.name] = s
			self.initializers[initializer.name] = onnx.numpy_helper.to_array(initializer)

	def loadInput(self, model:onnx.ModelProto):
		w = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
		h = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
		c = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value

		#w = 128
		#h = 96
		in_shape = IOShape(w,h,c)
		in_shape.name = model.graph.input[0].name
		self.ioshapes[in_shape.name] = in_shape

	def getLayer(self, name:str):
		for layer in self.layers:
			if layer.name == name:
				return layer
		return None

	# 检查每层输出是否被不同输入的层引用了
	def checkIOFormat(self):

		for layer in self.layers:
			if layer.type == "conv" and layer.ksize == 1:
				for input in layer.inputs:
					input.format = "cwh"
			elif (layer.type == "conv" and layer.ksize == 3 and (layer.inc == 1 or layer.group > 1)) or layer.type in ["maxp", "upsample", "global_avg_pool"]:
				for input in layer.inputs:
					input.format = "whc"

		for idx in range(0, len(self.layers)):
			layer = self.layers[idx]
			if layer.idx == 51:
				print()

			layer.outputs[0].ref = 0
			for idx2 in range(idx+1, len(self.layers)):
				after = self.layers[idx2]

				for input in after.inputs:
					prev_layer:Layer = self.layer_out_dict.get(input.name)
					if prev_layer == layer:						
						if layer.outputs[0].format == "any":
							layer.outputs[0].format = after.inputs[0].format
							if layer.outputs[0].ref < after.idx:
								layer.outputs[0].ref = after.idx
						else:
							found = False
							for output in layer.outputs:
								if output.format == after.inputs[0].format:
									if (output.ref < after.idx):
										output.ref = after.idx
									found = True
									break
							if not found and after.inputs[0].format != "any":
								newout = copy.copy(layer.outputs[0])									
								newout.format = after.inputs[0].format
								newout.name = layer.outputs[0].name + "-" + newout.format
								newout.ref = after.idx									
								layer.outputs.append(newout)
								self.layer_out_dict[newout.name] = layer

		#for layer in self.layers:
		#	if len(layer.inputs) > 0 and layer.inputs[0].format == "":
		#		for input in layer.inputs:
		#			input.format = layer.outputs[0].format
		#	if layer.outputs[0].format == "":
		#		layer.outputs[0].format = layer.inputs[0].format

	def loadConstant(self, model:onnx.ModelProto):
		for node in model.graph.node:
			if node.op_type == "Constant":
				for attr in node.attribute:
					if attr.name == "value":
						self.constant_values[node.output[0]] = onnx.numpy_helper.to_array(attr.t)

	def setLayerInput(self, layer:Layer, input:IOShape, name):
		channel = copy.copy(input)

		off_c = self.gathers.get(name)
		if off_c is not None:				
			channel.start_c = off_c * channel.c / 2
			channel.end_c = channel.start_c + channel.c / 2 - 1

		layer.inputs.append(channel)

	def loadNodes(self, model:onnx.ModelProto):
		self.layers.clear()
		for node in model.graph.node:
			#print(node.name)
			#print("node:", node.name, node.op_type)#, node.input, node.output)
			if self.end_names.count(node.name) > 0:
				continue

			layer = Layer()
			layer.name = node.name
			outc = 0
			ws:WeightShap = None #weight shape
			for name in node.input:
				shape = self.ioshapes.get(name)
				prev_layer:Layer = self.layer_out_dict.get(name)
				if node.op_type != "Reshape" and node.op_type != "Transpose":
					if prev_layer is not None and prev_layer.type == "concat":
						for ii in prev_layer.inputs:
							self.setLayerInput(layer, ii, name)
					elif shape is not None:
						self.setLayerInput(layer, shape, name)
				elif shape is not None:
					self.setLayerInput(layer, shape, name)
				# 卷积层的输出通道数根据偏移量的数量决定
				b = self.biases.get(name)
				if b is not None:
					outc = b
					layer.bias_name = name
				if self.weights.get(name) is not None:
					ws = self.weights.get(name)
					layer.weights_name = name

			if len(layer.inputs) == 0:
				continue
			#print("node:", node.name)
			# 计算输入尺寸
			layer.inw = layer.inputs[0].w
			layer.inh = layer.inputs[0].h
			layer.inc = 0
			for input in layer.inputs:
				layer.inc += input.end_c - input.start_c + 1
			
			if node.op_type == "Conv" or node.op_type == "MaxPool":
				if node.op_type == "Conv":
					layer.activate = "linear"
					layer.type = "conv"
				else:
					layer.type = "maxp"

				for attr in node.attribute:
					if attr.name == "group":
						layer.group = attr.i
					elif attr.name == "kernel_shape":
						layer.ksize = attr.ints[0]
					elif attr.name == "pads":
						layer.pad = attr.ints[0]
					elif attr.name == "strides":
						layer.stride = attr.ints[0]

				if outc == 0:
					outc = layer.inc

				outw = layer.inw / layer.stride
				outh = layer.inh / layer.stride
				output = IOShape(outw, outh, outc)
				output.name = node.output[0]
				layer.outputs.append(output)
				layer.idx = len(self.layers)

				if ws is not None:
					assert(layer.inc == ws.inc*layer.group)
					assert(outc == ws.outc)
					assert(layer.ksize == ws.kx)

				self.ioshapes[layer.outputs[0].name] = copy.copy(output)

			elif node.op_type == "Relu":
				prev_layer.outputs[0].name = node.output[0]
				prev_layer.activate = "relu"
				self.ioshapes[prev_layer.outputs[0].name] = copy.copy(prev_layer.outputs[0])
				self.layer_out_dict[prev_layer.outputs[0].name] = prev_layer
			elif node.op_type == "LeakyRelu":
				prev_layer.outputs[0].name = node.output[0]
				prev_layer.activate = "leaky"
				self.ioshapes[prev_layer.outputs[0].name] = copy.copy(prev_layer.outputs[0])
				self.layer_out_dict[prev_layer.outputs[0].name] = prev_layer
			elif node.op_type == "Sigmoid":
				prev_layer.outputs[0].name = node.output[0]
				prev_layer.activate = "sigmoid"
				self.ioshapes[prev_layer.outputs[0].name] = copy.copy(prev_layer.outputs[0])
				self.layer_out_dict[prev_layer.outputs[0].name] = prev_layer

			elif node.op_type == "Concat":
				# shuffle层开始
				layer.type = "concat"

				output = IOShape(layer.inw, layer.inh, layer.inc)
				output.name = node.output[0]
				layer.outputs.append(output)
				#layer.idx = len(self.layers)
				self.layer_out_dict[layer.outputs[0].name] = layer
				self.ioshapes[layer.outputs[0].name] = copy.copy(output)
			elif node.op_type == "Reshape" or node.op_type == "Transpose":
				prev_layer = self.layer_out_dict.get(layer.inputs[0].name)
				if node.op_type == "Reshape":
					prev_layer.type += "_reshape"
				else:
					prev_layer.type += "_transpos"

				if prev_layer.type == "concat_reshape_transpos_reshape":
					prev_layer.idx = len(self.layers)
					prev_layer.type = "shuffle"
					self.layers.append(prev_layer)

				prev_layer.outputs[0].name = node.output[0]
				self.ioshapes[prev_layer.outputs[0].name] = copy.copy(prev_layer.outputs[0])
				self.layer_out_dict[prev_layer.outputs[0].name] = prev_layer
			elif node.op_type == "Gather":
				prev_layer = self.layer_out_dict.get(layer.inputs[0].name)
				idx = self.constant_values.get(node.input[1]).item(0)
				#prev_layer.output.name = node.output[0]
				self.ioshapes[node.output[0]] = copy.copy(prev_layer.outputs[0])
				self.layer_out_dict[node.output[0]] = prev_layer
				self.gathers[node.output[0]] = idx

			elif node.op_type == "Resize":
				layer.type = "upsample"
				output = IOShape(layer.inw*2, layer.inh*2, layer.inc)
				output.name = node.output[0]
				layer.outputs.append(output)
				layer.ksize = 2
				layer.idx = len(self.layers)
				
				self.ioshapes[layer.outputs[0].name] = copy.copy(output)

			elif node.op_type == "Split":
				# 相当于2个gather
				prev_layer = self.layer_out_dict.get(layer.inputs[0].name)
				prev_output = prev_layer.outputs[0]
				output1 = IOShape(prev_output.w, prev_output.h, prev_output.c)
				output2 = IOShape(prev_output.w, prev_output.h, prev_output.c)
				output1.name = node.output[0]
				output2.name = node.output[1]
				self.ioshapes[output1.name] = output1
				self.ioshapes[output2.name] = output2
				self.layer_out_dict[output1.name] = prev_layer
				self.layer_out_dict[output2.name] = prev_layer
				self.gathers[output1.name] = 0
				self.gathers[output2.name] = 1

			elif node.op_type == "GlobalAveragePool":
				layer.type = "global_avg_pool"
				output = IOShape(1, 1, layer.inc)
				output.name = node.output[0]
				layer.outputs.append(output)
				layer.idx = len(self.layers)
				
				self.ioshapes[layer.outputs[0].name] = copy.copy(output)

			elif node.op_type == "Mul":
				layer.type = "mul"
				assert(len(layer.inputs) == 2)
				assert(layer.inputs[0].c == layer.inputs[1].c or layer.inputs[1].c == 1)
				assert(layer.inputs[1].w == 1 or layer.inputs[0].w == layer.inputs[1].w)
				assert(layer.inputs[1].h == 1 or layer.inputs[0].h == layer.inputs[1].h)
				layer.inc = layer.inputs[0].c
				outw = max(layer.inputs[0].w, layer.inputs[1].w)
				outh = max(layer.inputs[0].h, layer.inputs[1].h)
				outc = layer.inputs[0].c
				output = IOShape(outw, outh, outc)
				output.name = node.output[0]
				layer.outputs.append(output)
				layer.idx = len(self.layers)
				self.ioshapes[layer.outputs[0].name] = copy.copy(output)

			elif node.op_type == "Add":
				layer.type = "add"
				assert(len(layer.inputs) == 2)
				assert(layer.inputs[0].end_c - layer.inputs[0].start_c == layer.inputs[1].end_c - layer.inputs[1].start_c)
				assert(layer.inputs[0].w == layer.inputs[1].w)
				assert(layer.inputs[0].h == layer.inputs[1].h)
				layer.inc = layer.inputs[0].end_c - layer.inputs[0].start_c + 1
				outw =layer.inputs[0].w
				outh =layer.inputs[0].h
				outc = layer.inc
				output = IOShape(outw, outh, outc)
				output.name = node.output[0]
				layer.outputs.append(output)
				layer.idx = len(self.layers)
				self.ioshapes[layer.outputs[0].name] = copy.copy(output)

			elif node.op_type == "ReduceMax" or node.op_type == "ReduceMean":
				if node.op_type == "ReduceMax":
					layer.type = "reduce_max"
				else:
					layer.type = "reduce_mean"
				
				outw =layer.inputs[0].w
				outh =layer.inputs[0].h
				outc = 1
				output = IOShape(outw, outh, outc)
				output.name = node.output[0]
				layer.outputs.append(output)
				layer.idx = len(self.layers)
				self.ioshapes[layer.outputs[0].name] = copy.copy(output)



			self.appendLayer(layer)


	def appendLayer(self, layer:Layer):
		if layer.idx >= 0:
			self.layers.append(layer)
			self.layer_out_dict[layer.outputs[0].name] = layer
		else:
			self.temp_layers.append(layer)

	def writeNetwork(self, filename):
		file = open(filename, mode="w")
		max_type_len = 0
		max_activate_len = 0
		for layer in self.layers:
			if len(layer.type) > max_type_len:
				max_type_len = len(layer.type)
			if len(layer.activate) > max_activate_len:
				max_activate_len = len(layer.activate) 

		for layer in self.layers:
			#print(layer.outputs[0].name)
			#print("{\"",layer.outputs[0].name, "\", ", layer.idx,  "},", sep="")
			
			text = str.format("[{:3d}],", layer.idx)
			text += str.format("{:3d}*{:3d}*{:3d}->{:3d}*{:3d}*{:3d}, ", int(layer.inw), int(layer.inh), int(layer.inc), int(layer.outputs[0].w), int(layer.outputs[0].h), int(layer.outputs[0].c))

			text += layer.type + ", "
			blanks = max_type_len - len(layer.type)
			for i in range(blanks):
				text += " "
			
			text += layer.activate + ", "
			blanks = max_activate_len - len(layer.activate)
			for i in range(blanks):
				text += " "

			text += str.format("{:d},", layer.ksize)
			text += str.format("{:d},", layer.stride)
			text += str.format("{:d},", layer.pad)
			text += str.format("{:2d},", layer.group)

			for input in layer.inputs:
				prev_layer:Layer = self.layer_out_dict.get(input.name)				
				if prev_layer is not None:
					text += input.format + str.format("[{:3d}]{:2d}+{:2d};", prev_layer.idx, int(input.start_c), int(input.end_c - input.start_c + 1))
			text += ","

			for output in layer.outputs:
				text += output.format + " "
			text += ",\n"

			file.write(text)
		file.close()

	def writeLayerNames(self, filename):
		file = open(filename, mode="w")
		for layer in self.layers:
			file.write(layer.outputs[0].name.replace("/", "_") + "\n")
		file.close()

	def writeWeights(self, filename):
		file = open(filename, "w")
		kcount = 0
		bcount = 0
		for layer in self.layers:
			if layer.type == "conv":
				bias = self.initializers.get(layer.bias_name)
				weights = self.initializers.get(layer.weights_name)

				file.write("layer" + "{:.0f}".format(layer.idx) + "\n")

				iter = numpy.nditer(weights)
				kcount += iter.itersize
				file.write("kernel\n")
				for v in iter:
					file.write("{:.8f}, ".format(v))
				file.write("\n")

				iter = numpy.nditer(bias)
				bcount += iter.itersize
				file.write("bias\n")
				for v in iter:
					file.write("{:.8f}, ".format(v))
				file.write("\n")
		file.close()
		print("kcount:", kcount, "bcount:", bcount)

	def write8bitWeights(self, filename):
		file = open(filename, "w")
		kcount = 0
		bcount = 0
		for layer in self.layers:
			if layer.type == "conv":
				bias = self.initializers.get(layer.bias_name)
				weights = self.initializers.get(layer.weights_name)

				file.write("layer" + "{:.0f}".format(layer.idx) + "\n")

				iter = numpy.nditer(weights)
				kcount += iter.itersize
				file.write("kernel\n")
				for v in iter:
					file.write("{:.8f}, ".format(v))
				file.write("\n")

				iter = numpy.nditer(bias)
				bcount += iter.itersize
				file.write("bias\n")
				for v in iter:
					file.write("{:.8f}, ".format(v))
				file.write("\n")
		file.close()
		print("kcount:", kcount, "bcount:", bcount)