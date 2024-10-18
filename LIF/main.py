from pymonntorch import *
from exlif import EXLIF
from lif import LIF
from aelif import AELIF
from time_res import TimeResolution
from currents import ConstantCurrent, StepFunction, SineFunc, noiseFunc, shortPulse, LinearCurrent
from refractory import Refractory
import torch
import matplotlib.pyplot as plt


torch.manual_seed(73)

def A (model):
      net = Network(behavior={1: TimeResolution(dt=0.1)}, dtype= torch.float32, device='cpu', synapse_mode='SxD')
      pop1 = None
      if model == "lif" :
            pop1 = NeuronGroup(size=10,
                        net=net,
                        behavior={
                              2: ConstantCurrent(value=15),
                              5: LIF(R=1,
                                          thresh=-45,
                                          u_rest=-55,
                                          u_reset=-75,
                                          tau=10),
                              7: Recorder(tag="pop1_rec", variables=["u", "I", "freq"]),
                              8: EventRecorder(tag="pop1_event", variables=["spike"])})
            
      elif model == "exlif" :
            pop1 = NeuronGroup(size=10,
                        net=net,
                        behavior={
                              2: shortPulse(value=12, T=30),
                              5: EXLIF(R=5,
                                          thresh=-37,
                                          rh=-40,
                                          u_rest=-67,
                                          u_reset=-70,
                                          tau=10,
                                          delta=0.5
                                    ),
                              7: Recorder(tag="pop1_rec",variables=["u", "I", "freq"]),
                              8: EventRecorder(tag="pop1_event", variables=["spike"])})
            
      elif model == "aelif" :
            pop1 = NeuronGroup(size=10,
                        net=net,
                        behavior={
                              2: ConstantCurrent(value=12),
                              5: AELIF(R=5,
                                          thresh=-30,
                                          rh=-50,
                                          u_rest=-70,
                                          u_reset=-55,
                                          tauM=20,
                                          tauW=10,
                                          a=0,
                                          b=5,
                                          delta=2),
                              7: Recorder(tag="pop1_rec", variables=["u", "I", "freq"]),
                              8: EventRecorder(tag="pop1_event", variables=["spike"])})

      net.initialize()
      net.simulate_iterations(1000)
      plt.plot(net["u", 0][:,:1])
      plt.show()
      plt.plot(net["I", 0][:,:1])
      plt.show()      

def B (model, noise, s):
      net = Network(behavior={1: TimeResolution(dt=0.1)}, dtype= torch.float32, device='cpu', synapse_mode='SxD')
      pop1 = None
      if model == "lif" :
            pop1 = NeuronGroup(size=10,
                        net=net,
                        behavior={
                              2: noiseFunc(noise=noise, s=s),
                              5: LIF(R=1,
                                          thresh=-45,
                                          u_rest=-55,
                                          u_reset=-75,
                                          tau=10),
                              7: Recorder(tag="pop1_rec", variables=["u", "I", "freq"]),
                              8: EventRecorder(tag="pop1_event", variables=["spike"])})
            
      elif model == "exlif" :
            pop1 = NeuronGroup(size=10,
                        net=net,
                        behavior={
                              2: noiseFunc(noise=noise, s=s),
                              5: EXLIF(R=5,
                                          thresh=-37,
                                          rh=-40,
                                          u_rest=-67,
                                          u_reset=-70,
                                          tau=10,
                                          delta=0.5
                                    ),
                              7: Recorder(tag="pop1_rec",variables=["u", "I", "freq"]),
                              8: EventRecorder(tag="pop1_event", variables=["spike"])})
            
      elif model == "aelif" :
            pop1 = NeuronGroup(size=10,
                        net=net,
                        behavior={
                              2: noiseFunc(noise=noise, s=s),
                              5: AELIF(R=5,
                                          thresh=-30,
                                          rh=-50,
                                          u_rest=-70,
                                          u_reset=-55,
                                          tauM=20,
                                          tauW=10,
                                          a=0,
                                          b=5,
                                          delta=2),
                              7: Recorder(tag="pop1_rec", variables=["u", "I", "freq"]),
                              8: EventRecorder(tag="pop1_event", variables=["spike"])})
      net.initialize()
      net.simulate_iterations(1000)
      plt.plot(net["u", 0][:,:1])
      plt.show()
      plt.plot(net["I", 0][:,:1])
      plt.show()

def C (model, currents):
      pop1 = None
      freqs = []
      for current in currents :
            net = Network(behavior={1: TimeResolution(dt=0.1)}, dtype= torch.float32, device='cpu', synapse_mode='SxD')
            if model == "lif" :
                  pop1 = NeuronGroup(size=10,
                              net=net,
                              behavior={
                                    2: ConstantCurrent(value=current),
                                    5: LIF(R=1,
                                                thresh=-45,
                                                u_rest=-55,
                                                u_reset=-75,
                                                tau=10),
                                    7: Recorder(tag="pop1_rec", variables=["u", "I", "freq"]),
                                    8: EventRecorder(tag="pop1_event", variables=["spike"])})
                  
            elif model == "exlif" :
                  pop1 = NeuronGroup(size=10,
                              net=net,
                              behavior={
                                    2: ConstantCurrent(value=current),
                                    5: EXLIF(R=5,
                                                thresh=-37,
                                                rh=-40,
                                                u_rest=-67,
                                                u_reset=-70,
                                                tau=10,
                                                delta=0.5
                                          ),
                                    7: Recorder(tag="pop1_rec",variables=["u", "I", "freq"]),
                                    8: EventRecorder(tag="pop1_event", variables=["spike"])})
                  
            elif model == "aelif" :
                  pop1 = NeuronGroup(size=10,
                              net=net,
                              behavior={
                                    2: ConstantCurrent(value=current),
                                    5: AELIF(R=5,
                                                thresh=-30,
                                                rh=-50,
                                                u_rest=-70,
                                                u_reset=-55,
                                                tauM=20,
                                                tauW=10,
                                                a=0,
                                                b=5,
                                                delta=2),
                                    7: Recorder(tag="pop1_rec", variables=["u", "I", "freq"]),
                                    8: EventRecorder(tag="pop1_event", variables=["spike"])})

            net.initialize()
            net.simulate_iterations(1000)
            freqs.append(net["freq", 0])

      plt.plot(currents, [row[-1:,:][0][0] for row in freqs]) # change 3rd [] for plotting ith neuron
      plt.show()

def D (model, refMode=None, refTime=None, threshold0=None, tauA=None, theta=None):
      net = Network(behavior={1: TimeResolution(dt=0.1)}, dtype= torch.float32, device='cpu', synapse_mode='SxD')
      pop1 = None
      if model == "lif" :
            pop1 = NeuronGroup(size=10,
                        net=net,
                        behavior={
                              2: ConstantCurrent(value=15),
                              5: LIF(R=1,
                                          thresh=-45,
                                          u_rest=-55,
                                          u_reset=-75,
                                          tau=10,
                                          refMode=refMode,
                                          refTime=refTime, 
                                          threshold0=threshold0, 
                                          tauA=tauA, 
                                          theta=theta),
                              7: Recorder(tag="pop1_rec", variables=["u", "I", "freq"]),
                              8: EventRecorder(tag="pop1_event", variables=["spike"])})
            
      elif model == "exlif" :
            pop1 = NeuronGroup(size=10,
                        net=net,
                        behavior={
                              2: shortPulse(value=12, T=30),
                              5: EXLIF(R=5,
                                          thresh=-37,
                                          rh=-40,
                                          u_rest=-67,
                                          u_reset=-70,
                                          tau=10,
                                          delta=0.5,
                                          refMode=refMode,
                                          refTime=refTime, 
                                          threshold0=threshold0, 
                                          tauA=tauA, 
                                          theta=theta),
                              7: Recorder(tag="pop1_rec",variables=["u", "I", "freq"]),
                              8: EventRecorder(tag="pop1_event", variables=["spike"])})
            
      elif model == "aelif" :
            pop1 = NeuronGroup(size=10,
                        net=net,
                        behavior={
                              2: ConstantCurrent(value=12),
                              5: AELIF(R=5,
                                          thresh=-30,
                                          rh=-50,
                                          u_rest=-70,
                                          u_reset=-55,
                                          tauM=20,
                                          tauW=10,
                                          a=0,
                                          b=5,
                                          delta=2,
                                          refMode=refMode,
                                          refTime=refTime, 
                                          threshold0=threshold0, 
                                          tauA=tauA, 
                                          theta=theta),
                              7: Recorder(tag="pop1_rec", variables=["u", "I", "freq"]),
                              8: EventRecorder(tag="pop1_event", variables=["spike"])})
            
      net.initialize()
      net.simulate_iterations(1000)
      plt.plot(net["u", 0][:,:1])
      plt.show()
      plt.plot(net["I", 0][:,:1])
      plt.show() 


# A("lif")

B("exlif", 2, 10)

# C("lif", [10,12,15,17,25,30])
      
# D("lif", refMode="ADAPTIVE_THRESHOLD", threshold0=-50, tauA=10, theta=1)
