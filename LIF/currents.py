from pymonntorch import *
import math


# class Current(Behavior):
# 	def initialize(self, ng):
# 		self.currentMode = self.parameter("MODE", default="ConstantCurrent")
# 		if 
		


class ConstantCurrent(Behavior):
    def initialize(self, ng):
        self.value = self.parameter("value", None)
        ng.I = ng.vector(self.value)

    def forward(self, ng):
        ng.I = ng.vector(self.value)


class LinearCurrent(Behavior):
    def initialize(self, ng):
        self.a = self.parameter("a", default=1)
        self.b = self.parameter("b", default=0)
        ng.I = ng.vector(mode = self.b)
    def forward(self, ng):
        ng.I += ng.vector(self.a*ng.network.dt)


class StepFunction(Behavior):
	def initialize(self, ng):
		self.value = self.parameter("value")
		self.t0 = self.parameter("t0",default=0)
		self.t1 = self.parameter("t1",default=10)
		ng.I = ng.vector(mode="zeros")

	def forward(self, ng):
		if self.t1 >= ng.network.iteration * ng.network.dt >= self.t0:
			ng.I = ng.vector(mode=self.value)
		else:
			ng.I = ng.vector(mode="zeros")
			

class SineFunc(Behavior):
	def initialize(self, ng: NeuronGroup):
		self.a = self.parameter("a", default=1)
		self.b = self.parameter("b", default=1)
		self.c = self.parameter("c", default=0)
		ng.I = ng.vector(mode=self.c)	

	def forward(self, ng: NeuronGroup):
			sinFunc = self.a*float(math.sin(self.b*ng.network.dt*ng.network.iteration)) + self.c
			ng.I = ng.vector(mode=sinFunc)


class noiseFunc(Behavior):
	def initialize(self, ng):
		self.s = self.parameter("s", default=4)
		self.noise = self.parameter("noise", default=4)
		ng.I = ng.vector(mode=self.s)

	def forward(self, ng):
		ng.I += (ng.vector(mode="uniform")*self.noise)-self.noise/2


class shortPulse(Behavior):
	def initialize(self, ng):
		self.value = self.parameter("value")
		self.t0 = self.parameter("t0", default=0)
		self.T = self.parameter("T", default=30)
		ng.I = ng.vector(mode="zeros")
		self.cnt = 0 

	def forward(self, ng):
		if ng.network.dt*ng.network.iteration >= self.t0 :
			self.cnt+=1
		if (ng.network.dt*ng.network.iteration >= self.t0) and (self.cnt % self.T == 0) :
			ng.I = ng.vector(mode=self.value)
		else:
			ng.I.fill_(0)

