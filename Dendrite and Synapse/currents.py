from pymonntorch import *
import math

class Current(Behavior):
	def initialize(self, ng):
		self.currentMode = self.parameter("MODE", default="ConstantCurrent")
		if self.currentMode == "ConstantCurrent" :
			self.value = self.parameter("value", None)
			ng.I = ng.vector(self.value)
			ng.inpI = ng.vector(mode=self.value)
		elif self.currentMode == "LinearCurrent" :
			self.a = self.parameter("a", default=1)
			self.b = self.parameter("b", default=0)
			ng.I = ng.vector(mode = self.b)
		elif self.currentMode == "StepCurrent" :
			self.value = self.parameter("value")
			self.t0 = self.parameter("t0",default=0)
			self.t1 = self.parameter("t1",default=10)
			self.t2 = self.parameter("t2",default=1000000)
			self.t3 = self.parameter("t3",default=1000000)
			ng.I = ng.vector(mode="zeros")
		elif self.currentMode == "SineCurrent" :
			self.a = self.parameter("a", default=1)
			self.b = self.parameter("b", default=1)
			self.c = self.parameter("c", default=0)
			ng.I = ng.vector(mode=self.c)
		elif self.currentMode == "NoiseCurrent" :
			self.s = self.parameter("s", default=4)
			self.noise = self.parameter("noise", default=4)
			self.t0 = self.parameter("t0",default=100000)
			ng.I = ng.vector(mode=self.s)
			ng.inpI = ng.vector(mode=self.s)
		elif self.currentMode == "ShortPulse" :
			self.value = self.parameter("value")
			self.t0 = self.parameter("t0", default=0)
			self.T = self.parameter("T", default=30)
			ng.I = ng.vector(mode="zeros")
			self.cnt = 0 

	def forward(self, ng):
		if self.currentMode == "ConstantCurrent" :
			ng.I = ng.vector(self.value)
			ng.inpI = ng.vector(self.value)
			# pass
		elif self.currentMode == "LinearCurrent" :
			ng.I += ng.vector(self.a*ng.network.dt)
			ng.inpI += ng.vector(self.a*ng.network.dt)
		elif self.currentMode == "StepCurrent" :   
			if (self.t1 >= ng.network.iteration * ng.network.dt >= self.t0) or (self.t3 >= ng.network.iteration * ng.network.dt >= self.t2):
				ng.I = ng.inpI+ng.vector(mode=self.value)
			else:
				ng.I += ng.inpI+ng.vector(mode="zeros")
		elif self.currentMode == "SineCurrent" :
			# not modified for connected pops
			sinFunc = self.a*float(math.sin(self.b*ng.network.dt*ng.network.iteration)) + self.c
			ng.I = ng.vector(mode=sinFunc)
		elif self.currentMode == "NoiseCurrent" :
			n = (ng.vector(mode="uniform")*self.noise)-self.noise/2
			ng.I = ng.inpI+n
			ng.inpI += n
			if (ng.network.iteration * ng.network.dt == self.t0):
				ng.I += 5
				ng.inpI +=5
		elif self.currentMode == "ShortPulse" :
			# not modified for connected pops
			if ng.network.dt*ng.network.iteration >= self.t0 :
				self.cnt+=1
			if (ng.network.dt*ng.network.iteration >= self.t0) and (self.cnt % self.T == 0) :
				ng.I = ng.vector(mode=self.value)
			else:
				ng.I.fill_(0)
