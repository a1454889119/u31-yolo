from enum import Enum
from copy import copy
class RamStatus(str, Enum):
	IN_USE = "IN_USE"
	FREE = "FREE"

class RamBlock:
	name = ""
	addr:int = 0
	len:int = 0
	status:RamStatus = RamStatus.FREE
	next = None
	prev = None

	def __init__(self, name:str, addr, len) -> None:
		self.addr = addr
		self.len = len
		self.name = name

	def removeFromList(self):
		if self.prev is not None:
			self.prev.next = self.next
		if self.next is not None:
			self.next.prev = self.prev

	def insertNodeBeforeSelf(self, node):
		prev = self.prev
		node.prev = prev
		node.next = self
		self.prev = node
		if prev is not None:
			prev.next = node

	def insertNodeAfterSelf(self, node):
		next = self.next
		node.prev = self
		node.next = next
		self.next = node
		if next is not None:
			next.prev = node


class RamManager:
	total_size:int = 0
	blocks:list[RamBlock] = []
	history = {str:RamBlock}

	def __init__(self, total_size:int) -> None:
		self.total_size = total_size
		self.history.clear()
		self.blocks.append(RamBlock("ALL_MEMORY", 0, total_size))

	def alloc(self, name:str, len:int) -> RamBlock:
		head = self.blocks[0]
		while head.prev is not None:
			head = head.prev
		
		blk = head
		while blk is not None:
			if blk.status == RamStatus.FREE and blk.len >= len:
				allocated = RamBlock(name, blk.addr, len)
				allocated.status = RamStatus.IN_USE
				blk.insertNodeBeforeSelf(allocated)

				blk.addr += len
				blk.len -= len
				if blk.len == 0:
					blk.removeFromList()
					self.blocks.remove(blk)
				self.blocks.append(allocated)
				self.history[allocated.name] = (copy(allocated))
				return allocated
			blk = blk.next			
		return None
	
	def allocBack(self, name:str, len:int) -> RamBlock:
		rear = self.blocks[0]
		while rear.next is not None:
			rear = rear.next

		blk = rear
		while blk is not None:
			if blk.status == RamStatus.FREE and blk.len >= len:
				allocated = RamBlock(name, blk.addr + blk.len - len, len)
				allocated.status = RamStatus.IN_USE
				blk.insertNodeAfterSelf(allocated)

				blk.len -= len
				if blk.len == 0:
					blk.removeFromList()
					self.blocks.remove(blk)
				self.blocks.append(allocated)
				self.history[allocated.name] = (copy(allocated))
				return allocated
			blk = blk.prev
		return None

	def free(self, addr:int) -> bool:
		for blk in self.blocks:
			if blk.status == RamStatus.IN_USE and blk.addr == addr:
				blk.status = RamStatus.FREE
				merged = False
				prev = blk.prev
				next = blk.next
				# 1. 如果前一个结点是free的, 则和前一个合并
				if prev is not None:
					if prev.status == RamStatus.FREE:
						prev.len += blk.len
						# 移除blk
						blk.removeFromList()
						self.blocks.remove(blk)
						merged = True

				# 2. 如果后一个结点是free的, 则和后一个合并
				if next is not None:
					if next.status == RamStatus.FREE:
						if merged:
							prev.len += blk.next.len							
						else:
							blk.len += blk.next.len							
							merged = True
						# 移除next
						next.removeFromList()
						self.blocks.remove(next)				

				return True
		return False
	
	def free_back(self, addr:int, len:int) -> bool:
		for blk in self.blocks:
			if blk.status == RamStatus.IN_USE and blk.addr == addr:
				assert len < blk.len
				blk.len -= len

				freeAddr = blk.addr + blk.len - len
				next = blk.next
				if next is not None and next.status == RamStatus.FREE:
					next.addr = blk.addr + blk.len
					next.len += len
				else:
					freeBlk = RamBlock(blk.name + ".free", blk.addr+blk.len, len)
					blk.insertNodeAfterSelf(freeBlk)
					self.blocks.append(freeBlk)
					self.history[freeBlk.name] = freeBlk
				return True
		return False

	
	def print(self):		
		head = self.head()

		while head is not None:
			print(head.name, head.addr, head.len, head.status)
			head = head.next

	def head(self)->RamBlock:
		head = self.blocks[0]
		
		while head.prev is not None:
			head = head.prev
		return head

	
		