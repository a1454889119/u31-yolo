

class IOShape:
	name = ""
	w = 0
	h = 0
	c = 0
	start_c = 0
	end_c = 0
	ref = 0
	format = "any"
	qscale = 1.0
	qzero_point = 0

	def __init__(self, w, h, c):
		self.name = ""
		self.w = w
		self.h = h
		self.c = c
		self.start_c = 0
		self.end_c = c - 1

	def size(self)->int:
		return int(self.w*self.h*(self.end_c-self.start_c+1)*2)
	
	def size8bit(self)->int:
		return int(self.w*self.h*(self.end_c-self.start_c+1))

class WeightShap:
	name = ""
	inc = 0
	outc = 0
	kx = 0
	ky = 0
	qscale = 1.0
	qzero_point = 0

class Layer:
	name = ""
	type = ""
	idx = -1
	ksize = 1
	stride = 1
	pad = 0
	group = 1
	activate = ""
	inputs:list[IOShape] = []
	outputs:list[IOShape] = []
	inw = 0
	inh = 0
	inc = 0

	weights_name = ""
	bias_name = ""
	def __init__(self) -> None:
		self.name = ""
		self.type = ""
		self.idx = -1
		self.ksize = 1
		self.stride = 1
		self.pad = 0
		self.group = 1
		self.activate = ""
		self.inputs = []
		self.outputs = []
		self.inw = 0
		self.inh = 0
		self.inc = 0

	def copy(self, other):
		self.name = other.name
		self.type = other.type		
		self.ksize = other.ksize
		self.stride = other.stride
		self.pad = other.pad
		self.group = other.group
		self.activate = other.activate
		self.inw = other.inw
		self.inh = other.inh
		self.inc = other.inc
		self.weights_name = other.weights_name
		self.bias_name = other.bias_name
