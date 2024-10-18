from pymonntorch import *
import torch
import math

class InputForward(Behavior):
    def initialize(self, synapse: SynapseGroup):
        self.type = self.parameter("type", default = "exc")
        self.preN = synapse.matrix_dim()[0]
        self.postN = synapse.matrix_dim()[1]
        self.mode = self.parameter("MODE", default=None)
        self.coef = self.parameter("coef", None)
        if self.mode == "full":
            self.j = self.parameter("j", None)
            self.W = synapse.matrix(mode = self.j/(self.preN))
        elif self.mode == "fixedProb":
            self.p = self.parameter("p", None)
            self.j = self.parameter("j", None)
            prob = torch.rand(self.preN, self.postN)
            prob[prob <= (1-self.p)] = 0
            prob[prob > (1-self.p)] = 1
            self.W = synapse.matrix(mode = self.j/(self.preN*self.p))
            self.W *= prob
        elif self.mode == "fixedPartners":
            self.C = self.parameter("C", None)
            self.W = synapse.matrix(mode = "uniform")
            prob = torch.zeros(self.preN, self.postN)
            for col in range(self.postN):
                ones_positions = torch.randperm(self.preN)[:self.C]
                prob[ones_positions, col] = 1
            self.W *= prob
        elif self.mode == "balanced":
            self.p = self.parameter("p", None)
            self.j = self.parameter("j", None)
            self.W = synapse.matrix(mode = self.j/math.sqrt(self.preN*self.p))
        else:
            self.W = synapse.matrix(mode = "uniform")

        

    def forward(self, synapse):
        pre_spike = synapse.src.spike
        synapse.I = torch.sum(self.W[pre_spike]) * (1 if self.type=="exc" else -1) * self.coef
    
        