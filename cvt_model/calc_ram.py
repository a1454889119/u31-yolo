import onnx
from OnnxModelExport import Exporter, Layer, IOShape
from RamManager import *

# 避免内存碎片
not_cover_layer = [] # 强制不覆盖输入的层
alloc_back_layer = [11] # 强制用allocBack申请内存的层
# 输出层
ref48_layer = [43, 48]
ref64_layer = [59, 64]

total_memory = (256+16)*1024
total_memory -= 1024*20 # 程序20k
total_memory -= 76240 # 卷积核
total_memory -= 32*6*4+4 # 最大32个结果(每个结果6个32位整数), 1个4字节存储个数

ram = RamManager(total_memory)
imgblk = ram.allocBack("image", 160*120)

mm = onnx.load("../yolo-c/data/20250804_opsram_test_last_reordered_model.onnx")
exp = Exporter()
exp.end_names = ["/model.15/Concat", "/model.15/Concat_1"]
exp.loadModel(mm)

inputAddrs = {}
outputAddrs = {}

allocated = {imgblk.name: imgblk}

long_term_output = []
short_term_output = []
for layer in exp.layers:
	for output in layer.outputs:
		text = str.format("\tlayer {:2d} output ({:s},{:d}) can be freed at {:d})", layer.idx, output.format, output.size(), output.ref)
		if output.ref == layer.idx + 1:
			short_term_output.append(text)
		else:			
			long_term_output.append(text)

print("\n\nlong term outputs:\n\n")
for text in long_term_output:
	print(text)

print("big inputs:")
for layer in exp.layers:
	if len(layer.inputs) > 1:
		insize = int(layer.inw*layer.inh*layer.inc*2)
		text = str.format("\tlayer {:2d}({:s}) input size({:d}): ", layer.idx, layer.type, insize)
		for input in layer.inputs:
			prev_layer:Layer = exp.layer_out_dict.get(input.name)
			text += str.format("{:d}({:d}) ", prev_layer.idx, input.size())
		print(text)

with open("../160x120/memory_usage--.txt", "w") as usageFile, open("../160x120/memory--.upinc", "w") as headFile:
	headFile.write("\n@import MEMORY\n\n")
	# 这里的层输出的ref, 这里表示最大的引用层idx
	for idx in range(0, len(exp.layers)):
		layer = exp.layers[idx]
		# 0层之后释放图像输入
		if idx == 1:
			ram.free(imgblk.addr)
			allocated.pop(imgblk.name)
		
		for output in layer.outputs:
			if output.ref == 0:
				if layer.idx in ref48_layer:
					output.ref = 48
				elif layer.idx in ref64_layer:
					output.ref = 64

		#print(layer.idx, layer.output.ref)

		outsize = layer.outputs[0].w*layer.outputs[0].h*layer.outputs[0].c*2
		covered = False

		# 检查输出是否能覆盖输入
		if layer.idx not in not_cover_layer:
			informat = layer.inputs[0].format

			for output in layer.outputs:
				if informat == "any" or informat == output.format:
					for input in layer.inputs:
						insize = input.w*input.h*input.c*2
						prev_layer:Layer = exp.layer_out_dict.get(input.name)
						if prev_layer is not None:
							for poutput in prev_layer.outputs:
								if poutput.format == informat and poutput.ref == layer.idx:
									# 可以覆盖
									blk:RamBlock = allocated.get(output.name)
									if blk is None:
										continue
									
									if (insize - outsize > 0):
										ram.free_back(blk.addr, insize - outsize)
									allocated.pop(blk.name)
									blk.name = output.name
									allocated[blk.name] = blk
									outputAddrs[blk.name] = int(blk.addr)
									covered = True
									break
					if covered:
						break

										
				# 不能覆盖, 则申请内存
				if covered == False:
					blk:RamBlock = None
					if output.ref == layer.idx + 1 and layer.idx not in alloc_back_layer:
						blk = ram.alloc(output.name, outsize)
					else:
						blk = ram.allocBack(output.name, outsize)
					if (blk is None):
						ram.print()
						assert(0)
					allocated[blk.name] = blk
					outputAddrs[blk.name] = int(blk.addr)

		
		free:int = 0
		# 计算可用内存
		blk = ram.head()
		while blk is not None:
			if blk.status == RamStatus.FREE:
				free += blk.len

			blk = blk.next

		# all 和 addr 用于检测内存完整性
		all = 0 
		addr = 0
		# 输出ram现状
		text = "-------------layer"
		text += str.format("{:d}, {:d} free bytes, output name:", layer.idx, int(free))
		text += layer.outputs[0].name + "\n"
		usageFile.write(text)
		print(text, end="")
		blk = ram.head()
		while blk is not None:
			l:Layer = exp.layer_out_dict.get(blk.name)
			text = ""
			if l is not None:
				text += str.format("[{:d}] ", l.idx)
			else:
				text += blk.name + " "
			text += str.format("{:d} {:d} ", int(blk.addr), int(blk.len)) + blk.status
			
			if l is not None and blk.status == RamStatus.IN_USE:
				for output in l.outputs:
					if output.name == blk.name:
						text += str.format("(free after [{:d}])\n", output.ref) 
			else:
				text += "\n"

			print(text, end="")
			usageFile.write(text)
			if addr != blk.addr:
				assert(0)
			addr += blk.len

			all += blk.len
			blk = blk.next
		if all != total_memory:
			assert(0)

		# 检查可释放的内存
		remove_names = []
		for blk in allocated.values():
			prev_layer:Layer = exp.layer_out_dict.get(blk.name)		
			if prev_layer is not None:
				for output in prev_layer.outputs:
					if output.name == blk.name and output.ref <= layer.idx:	
						ram.free(blk.addr)
						remove_names.append(blk.name)

		for name in remove_names:
			allocated.pop(name)
	
	for name in outputAddrs.keys():
		layer:Layer = exp.layer_out_dict.get(name)
		if layer.idx > 0:
			for i in range(0, len(layer.inputs)):
				input = layer.inputs[i]
				l:Layer = exp.layer_out_dict.get(input.name)
				inAddr:int = outputAddrs.get(l.output.name)
				assert(inAddr is not None)
				inAddr += int(input.w * input.h * input.start_c)

				text = "#define LAYER_" + str(layer.idx) + "_INPUT"
				if len(layer.inputs) > 1:
					text += "_PART_" + str(i)
				text += " (MEMORY + " + str(inAddr) + ")\n"
				headFile.write(text)

		headFile.write("#define LAYER_" + str(layer.idx) + "_OUTPUT (MEMORY + " + str(outputAddrs[name]) + ")\n")
		headFile.write("\n")








	


	





	

                
	

