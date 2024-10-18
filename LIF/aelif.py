from pymonntorch import *
import torch

class AELIF(Behavior):
    def initialize(self, ng: NeuronGroup):
        self.R = self.parameter("R", None)
        self.tauM = self.parameter("tauM", None)
        self.tauW = self.parameter("tauW", None)
        self.u_rest = self.parameter("u_rest", None)
        self.u_reset = self.parameter("u_reset", None)
        self.threshold = self.parameter("threshold", None)
        self.rh = self.parameter("rh", None)
        self.delta = self.parameter("delta", None)
        self.a = self.parameter("a", None)
        self.b = self.parameter("b", None)

        self.refMode = self.parameter("refMode", None)
        self.refTime = self.parameter("refTime", default=0)
        self.threshold0 = self.parameter("threshold0", None)
        self.tauA = self.parameter("tauA", None)
        self.theta = self.parameter("theta", None)

        ng.threshold = ng.vector(mode=self.parameter("thresh"))
        ng.u = ng.vector(mode = self.parameter("u_init", default="normal(-75, 10)"))
        ng.w = ng.vector(mode = "zeros")
        ng.freq = ng.vector(mode="zeros")
        ng.lastSpike = ng.vector(mode="zeros")
        ng.spike = ng.u >= ng.threshold
        ng.spikeN = ng.spike.float()
        ng.u[ng.spike] = self.u_reset

    def forward(self, ng: NeuronGroup):
        if self.refMode == "BLOCK_CURRENT":
            refCondition = (ng.network.iteration*ng.network.dt - ng.lastSpike) < self.refTime
            leakage = -(ng.u - self.u_rest)
            inp_u = self.R * ng.I 
            inp_u[refCondition] = 0
            exp_term = self.delta * torch.exp((ng.u - self.rh)/(self.delta))
            ng.w += (((self.a*(ng.u-self.u_rest) - ng.w + self.b * self.tauW * ng.spikeN.float())) / self.tauW) * ng.network.dt
            ng.u += ((leakage + exp_term - self.R*ng.w + inp_u) / self.tauM) * ng.network.dt

        elif self.refMode == "ADAPTIVE_THRESHOLD":
            leakage = -(ng.u - self.u_rest)
            inp_u = self.R * ng.I 
            exp_term = self.delta * torch.exp((ng.u - self.rh)/(self.delta))
            ng.w += (((self.a*(ng.u-self.u_rest) - ng.w + self.b * self.tauW * ng.spikeN.float())) / self.tauW) * ng.network.dt
            ng.u += ((leakage + exp_term - self.R*ng.w + inp_u) / self.tauM) * ng.network.dt
            ng.threshold += ((-(ng.threshold - self.threshold0) + self.theta * ng.spikeN.float()) / self.tauA)*ng.network.dt

        else:
            leakage = -(ng.u - self.u_rest)
            inp_u = self.R * ng.I 
            exp_term = self.delta * torch.exp((ng.u - self.rh)/(self.delta))
            ng.w += (((self.a*(ng.u-self.u_rest) - ng.w + self.b * self.tauW * ng.spikeN.float())) / self.tauW) * ng.network.dt
            ng.u += ((leakage + exp_term - self.R*ng.w + inp_u) / self.tauM) * ng.network.dt

        ng.spike = ng.u >= ng.threshold
        ng.spikeN += ng.spike.float()
        ng.freq[ng.spike] = ng.spikeN[ng.spike]/(ng.network.dt*ng.network.iteration)
        ng.lastSpike[ng.spike] = ng.network.dt*ng.network.iteration
        ng.u[ng.spike] = self.u_reset