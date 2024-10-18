from pymonntorch import *
import math
import matplotlib.pyplot as plt
from lif import LIF
from time_res import TimeResolution
from currents import Current
from syn import InputForward
from dendrite import Dendrite
import torch
from scipy.interpolate import make_interp_spline

torch.manual_seed(73)

def A (couplingPars,modelParameters1, modelParameters2 ,currentMode, currentParameters1, currentParameters2, n=10):
        net = Network(behavior={1: TimeResolution(dt=0.1)}, dtype= torch.float32, device='cpu', synapse_mode='SxD')
        pop1 = NeuronGroup(size=n,
                net=net,
                behavior={
                        2: Current(**currentParameters1, MODE=currentMode),
                        # 4: Dendrite(),
                        5: LIF(**modelParameters1),
                        7: Recorder(tag="pop1_rec", variables=["u", "I"]),
                        8: EventRecorder(tag="pop1_event", variables=["spike"])})
        
        pop2 = NeuronGroup(size=n,
                net=net,
                behavior={
                        2: Current(**currentParameters2, MODE=currentMode),
                        4: Dendrite(),
                        5: LIF(**modelParameters2),
                        7: Recorder(tag="pop2_rec", variables=["u", "I"]),
                        8: EventRecorder(tag="pop2_event", variables=["spike"])})
        
        syn = SynapseGroup(tag="normal", net=net, src=pop1, dst=pop2, behavior={3: InputForward(**couplingPars)})

        net.initialize()
        net.simulate_iterations(1000)

        plt.figure().set_size_inches(7,2)
        plt.plot(net["pop1_rec", 0].variables["u"][:,:2])
        plt.xlabel("t")
        plt.ylabel("u")
        plt.title("Presynaptic population membrane potential")
        plt.show()

        plt.figure().set_size_inches(7,2)
        plt.plot(net["pop2_rec", 0].variables["u"][:,:2])
        plt.xlabel("t")
        plt.ylabel("u")
        plt.title("Postsynaptic population membrane potential")
        plt.show()

        plt.figure().set_size_inches(7,2)
        plt.plot(net["pop1_rec", 0].variables["I"][:,:1])
        plt.xlabel("t")
        plt.ylabel("I")
        plt.title("Presynaptic population Input current")
        plt.show()

        plt.figure().set_size_inches(7,2)
        plt.plot(net["pop2_rec", 0].variables["I"][:,:1])
        plt.xlabel("t")
        plt.ylabel("I")
        plt.title("Postsynaptic population Input current")
        plt.show()
        
def B (couplingMode, couplingPars, modelParameters1, modelParameters2 ,currentMode, currentParameters1, currentParameters2 ,n=10):
        net = Network(behavior={1: TimeResolution(dt=0.1)}, dtype= torch.float32, device='cpu', synapse_mode='SxD')
        pop1 = NeuronGroup(size=n,
                net=net,
                behavior={
                        2: Current(**currentParameters1, MODE=currentMode),
                        4: Dendrite(),
                        5: LIF(**modelParameters1),
                        7: Recorder(tag="pop1_rec", variables=["u", "I"]),
                        8: EventRecorder(tag="pop1_event", variables=["spike"])})

        pop2 = NeuronGroup(size=n,
                net=net,
                behavior={
                        2: Current(**currentParameters2, MODE=currentMode),
                        4: Dendrite(),
                        5: LIF(**modelParameters2),
                        7: Recorder(tag="pop2_rec", variables=["u", "I"]),
                        8: EventRecorder(tag="pop2_event", variables=["spike"])})
                
        syn = SynapseGroup(tag="normal", net=net, src=pop1, dst=pop2, behavior={3: InputForward(**couplingPars,MODE=couplingMode)})
        net.initialize()
        net.simulate_iterations(1000)
        plt.figure().set_size_inches(7,2)
        plt.plot(net["pop1_rec", 0].variables["u"][:,:3])
        plt.xlabel("t")
        plt.ylabel("u")
        plt.title("Presynaptic population membrane potential")
        plt.show()
        
        plt.figure().set_size_inches(7,2)
        plt.plot(net["pop2_rec", 0].variables["u"][:,:3])
        plt.xlabel("t")
        plt.ylabel("u")
        plt.title("Postsynaptic population membrane potential")
        plt.show()

        plt.figure().set_size_inches(7,2)
        plt.plot(net["pop1_rec", 0].variables["I"][:,:1])
        plt.xlabel("t")
        plt.ylabel("I")
        plt.title("Presynaptic population Input current")
        plt.show()

        plt.figure().set_size_inches(7,2)
        plt.plot(net["pop2_rec", 0].variables["I"][:,:1])
        plt.xlabel("t")
        plt.ylabel("I")
        plt.title("Postsynaptic population Input current")
        plt.show()

def C (couplingMode, couplingPars1, couplingPars2, modelParameters1, modelParameters2 ,currentMode1, currentMode2, currentParameters1, currentParameters2 ,n=10):
        net = Network(behavior={1: TimeResolution(dt=0.1)}, dtype= torch.float32, device='cpu', synapse_mode='SxD')
        pop1 = NeuronGroup(size=math.ceil(n*0.8),
                net=net,
                behavior={
                        2: Current(**currentParameters1, MODE=currentMode1),
                        4: Dendrite(),
                        5: LIF(**modelParameters1),
                        7: Recorder(tag="pop1_rec", variables=["u", "I", "A", "inpI"]),
                        8: EventRecorder(tag="pop1_event", variables=["spike"])})

        pop2 = NeuronGroup(size=math.ceil(n*0.2),
                net=net,
                behavior={
                        2: Current(**currentParameters2, MODE=currentMode2),
                        4: Dendrite(),
                        5: LIF(**modelParameters2),
                        7: Recorder(tag="pop2_rec", variables=["u", "I", "A", "inpI"]),
                        8: EventRecorder(tag="pop2_event", variables=["spike"])})
                
        syn12 = SynapseGroup(tag="normal12", net=net, src=pop1, dst=pop2, behavior={3: InputForward(**couplingPars1,MODE=couplingMode)})
        syn21 = SynapseGroup(tag="normal21", net=net, src=pop2, dst=pop1, behavior={3: InputForward(**couplingPars2,MODE=couplingMode)})

        net.initialize()
        net.simulate_iterations(1000)
        plt.figure().set_size_inches(7,2)
        plt.scatter(net["pop1_event",0]["spike",0][:,0],net["pop1_event",0]["spike",0][:,1], c='tab:blue', label = 'exc',s=3)
        plt.scatter(net["pop2_event",0]["spike",0][:,0],net["pop2_event",0]["spike",0][:,1]+math.ceil(n*0.8), c='tab:orange', label = 'inh',s=3)
        plt.title("Raster Plot")
        plt.legend()
        plt.show()
        plt.figure().set_size_inches(7,2)
        plt.plot(net["pop1_rec", 0].variables["A"])
        plt.title("EXC population Activity")
        plt.show()
        plt.figure().set_size_inches(7,2)
        plt.plot(net["pop2_rec", 0].variables["A"])
        plt.title("INH population Activity")
        plt.show()

        plt.figure().set_size_inches(7,2)
        plt.plot(net["pop1_rec", 0].variables["inpI"][:,:1])
        plt.xlabel("t")
        plt.ylabel("I")
        plt.title("EXC population Input current")
        plt.show()

        plt.figure().set_size_inches(7,2)
        plt.plot(net["pop2_rec", 0].variables["inpI"][:,:1])
        plt.xlabel("t")
        plt.ylabel("I")
        plt.title("INH population Input current")
        plt.show()

        plt.figure().set_size_inches(7,2)
        plt.plot(net["pop1_rec", 0].variables["I"][:,:1])
        plt.xlabel("t")
        plt.ylabel("I")
        plt.title("EXC population Total current")
        plt.show()

        plt.figure().set_size_inches(7,2)
        plt.plot(net["pop2_rec", 0].variables["I"][:,:1])
        plt.xlabel("t")
        plt.ylabel("I")
        plt.title("INH population Total current")
        plt.show()

def D(couplingMode, couplingParsEE ,couplingParsEI, couplingParsIE, modelParameters,currentMode, currentParameters1, currentParameters2, currentParameters3 ,n=10):
        net = Network(behavior={1: TimeResolution(dt=0.1)}, dtype= torch.float32, device='cpu', synapse_mode='SxD')
        pop1 = NeuronGroup(size=n,
                net=net,
                behavior={
                        2: Current(**currentParameters1, MODE=currentMode),
                        4: Dendrite(),
                        5: LIF(**modelParameters),
                        7: Recorder(tag="pop1_rec", variables=["u", "I", "A", "inpI"]),
                        8: EventRecorder(tag="pop1_event", variables=["spike"])})

        pop2 = NeuronGroup(size=n,
                net=net,
                behavior={
                        2: Current(**currentParameters2, MODE=currentMode),
                        4: Dendrite(),
                        5: LIF(**modelParameters),
                        7: Recorder(tag="pop2_rec", variables=["u", "I", "A", "inpI"]),
                        8: EventRecorder(tag="pop2_event", variables=["spike"])})
        
        pop3 = NeuronGroup(size=n,
                net=net,
                behavior={
                        2: Current(**currentParameters3, MODE=currentMode),
                        4: Dendrite(),
                        5: LIF(**modelParameters),
                        7: Recorder(tag="pop3_rec", variables=["u", "I", "A", "inpI"]),
                        8: EventRecorder(tag="pop3_event", variables=["spike"])})
                
        syn13 = SynapseGroup(tag="normal13", net=net, src=pop1, dst=pop3, behavior={3: InputForward(**couplingParsEI,MODE=couplingMode)})
        syn23 = SynapseGroup(tag="normal23", net=net, src=pop2, dst=pop3, behavior={3: InputForward(**couplingParsEI,MODE=couplingMode)})
        syn31 = SynapseGroup(tag="normal31", net=net, src=pop3, dst=pop1, behavior={3: InputForward(**couplingParsIE,MODE=couplingMode)})
        syn32 = SynapseGroup(tag="normal32", net=net, src=pop3, dst=pop2, behavior={3: InputForward(**couplingParsIE,MODE=couplingMode)})
        syn11 = SynapseGroup(tag="normal11", net=net, src=pop1, dst=pop1, behavior={3: InputForward(**couplingParsEE,MODE=couplingMode)})
        syn22 = SynapseGroup(tag="normal22", net=net, src=pop2, dst=pop2, behavior={3: InputForward(**couplingParsEE,MODE=couplingMode)})


        
        net.initialize()
        net.simulate_iterations(1000)
        plt.figure().set_size_inches(7,2)
        plt.scatter(net["pop1_event",0]["spike",0][:,0],net["pop1_event",0]["spike",0][:,1], label = 'exc1',s=1)
        # plt.title("Raster Plot EXC1")
        # plt.show()
        # plt.figure().set_size_inches(7,2)
        plt.scatter(net["pop2_event",0]["spike",0][:,0],net["pop2_event",0]["spike",0][:,1]+n, label = 'exc2',s=1)
        # plt.title("Raster Plot EXC2")
        # plt.show()
        # plt.figure().set_size_inches(7,2)
        plt.scatter(net["pop3_event",0]["spike",0][:,0],net["pop3_event",0]["spike",0][:,1]+2*n, label = 'inh',s=1)
        plt.title("Raster Plot")
        plt.legend()
        plt.show()
        
        plt.figure().set_size_inches(7,2)
        xnew = np.linspace(0,1000,300)
        spl1 = make_interp_spline(range(1000),net["pop1_rec", 0].variables["A"], k=3)
        smooooth1 = spl1(xnew)
        spl2 = make_interp_spline(range(1000),net["pop2_rec", 0].variables["A"], k=3)
        smooooth2 = spl2(xnew)
        spl3 = make_interp_spline(range(1000),net["pop3_rec", 0].variables["A"], k=3)
        smooooth3 = spl3(xnew)
        plt.plot(xnew[:-1], smooooth3[:-1], label= 'INH', c="green")
        plt.plot(xnew[:-1], smooooth1[:-1], label='EXC1')
        plt.plot(xnew[:-1], smooooth2[:-1], label = 'EXC2', c="orange")
        plt.title("Activity")
        plt.legend()
        plt.xlabel("t")
        plt.ylabel("A")
        plt.show()


        plt.figure().set_size_inches(7,2)
        xnew = np.linspace(0,1000,300)
        spl1 = make_interp_spline(range(1000),net["pop1_rec", 0].variables["I"][:,:1], k=5)
        smooooth1 = spl1(xnew)
        spl2 = make_interp_spline(range(1000),net["pop2_rec", 0].variables["I"][:,:1], k=5)
        smooooth2 = spl2(xnew)
        # spl3 = make_interp_spline(range(1000),net["pop3_rec", 0].variables["I"][:,:1], k=3)
        # smooooth3 = spl3(xnew)
        plt.plot(xnew[:-1], smooooth1[:-1], label='EXC1')
        plt.plot(xnew[:-1], smooooth2[:-1], label = 'EXC2')
        # plt.plot(xnew[:-1], smooooth3[:-1], label= 'INH')
        plt.title("Total input current")
        plt.xlabel("t")
        plt.ylabel("I")
        plt.legend()
        plt.show()


        # plt.plot(net["pop1_rec", 0].variables["A"])
        # plt.title("EXC1 Activity")
        # plt.show()
        # plt.figure().set_size_inches(7,2)
        # plt.plot(net["pop2_rec", 0].variables["A"])
        # plt.title("EXC2 Activity")
        # plt.show()
        # plt.figure().set_size_inches(7,2)
        # plt.plot(net["pop2_rec", 0].variables["A"])
        # plt.title("INH Activity")
        # plt.show()

        plt.figure().set_size_inches(7,2)
        plt.plot(net["pop1_rec", 0].variables["inpI"][:,:1], label='EXC1')
        plt.plot(net["pop2_rec", 0].variables["inpI"][:,:1], label='EXC2')
        plt.plot(net["pop3_rec", 0].variables["inpI"][:,:1], label='INH')
        plt.xlabel("t")
        plt.ylabel("I")
        plt.title("Input current")
        plt.legend()
        plt.show()

        # plt.figure().set_size_inches(7,2)
        # plt.plot(net["pop2_rec", 0].variables["inpI"][:,:1])
        # plt.xlabel("t")
        # plt.ylabel("I")
        # plt.title("EXC2 population Input current")
        # plt.show()




lifPars1 = {"R":5, "thresh":-37, "u_rest":-67, "u_reset":-75, "tau":10}
lifPars2 = {"R":5, "thresh":-37, "u_rest":-67, "u_reset":-75, "tau":10}
lifPars3 = {"R":5, "thresh":"normal(-40, 10)", "u_rest":-67, "u_reset":-75, "tau":10}

fullPars1 = {"j":2, "coef":0.25, "type":"exc"}
fullPars2 = {"j":5, "coef":0.25, "type":"inh"}

fixedProbPars1 = {"j":2, "p":0.2 ,"coef":0.25, "type":"exc"}
fixedProbPars2 = {"j":3, "p":0.2 ,"coef":0.25, "type":"inh"}

fixedPartnerPars1 = {"C":60, "coef":0.25, "type":"exc"}
fixedPartnerPars2 = {"C":60 ,"coef":0.25,"type":"inh"}


balancedPars1 = {"j":1, "p":0.2 ,"coef":0.5, "type":"exc"}
balancedPars2 = {"j":0.1, "p":0.2 ,"coef":0.5,"type":"inh"}

noisePars1 = {"s":7, "noise":0.2, "t0":50}
noisePars2 = {"s":7, "noise":0.2}
noisePars3 = {"s":2, "noise":0.2}

constantPars1 = {"value":5}
constantPars2 = {"value":5}
constantPars3 = {"value":4}


# A({"coef":100},lifPars1, lifPars2, "ConstantCurrent", {"value":7}, {"value":0})

# B("full", {"j":2}, lifPars1, lifPars2, "ConstantCurrent", {"value":7}, {"value":7})
# B("fixedProb", {"j":1, "p":0.1 ,"coef":1, "type":"exc"}, lifPars1, lifPars2, "NoiseCurrent", {"s":5, "noise":0.1}, {"s":0, "noise":0} , n=10)


# C("full", fullPars1, fullPars2, lifPars3, lifPars3,"NoiseCurrent","NoiseCurrent", noisePars2, noisePars3,  n=1000)
# C("fixedProb", fixedProbPars1, fixedProbPars2, lifPars3, lifPars3,"NoiseCurrent","NoiseCurrent", noisePars1, noisePars2,  n=1000)
# C("fixedPartners", fixedPartnerPars1, fixedPartnerPars2, lifPars3, lifPars3,"NoiseCurrent","NoiseCurrent", noisePars1, noisePars2,  n=100)
# C("balanced", balancedPars1, balancedPars2, lifPars3, lifPars3,"NoiseCurrent","NoiseCurrent", noisePars1, noisePars2,  n=100)


# D("full",fullPars1 ,fullPars1,fullPars2, lifPars3, "NoiseCurrent", noisePars1, noisePars2, noisePars3, 1000)
# D("full",fullPars1 ,fullPars1,fullPars2, lifPars3, "ConstantCurrent", constantPars1, constantPars2, constantPars3, 100)
# D("fixedProb",fixedProbPars1 ,fixedProbPars1,fixedProbPars2, lifPars3, "NoiseCurrent", noisePars1, noisePars2, noisePars3, 1000)
# D("fixedPartners",fixedPartnerPars1 ,fixedPartnerPars1,fixedPartnerPars2, lifPars3, "NoiseCurrent", noisePars1, noisePars2, noisePars3, 1000)
