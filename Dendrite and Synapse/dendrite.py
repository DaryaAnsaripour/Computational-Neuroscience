from pymonntorch import *

class Dendrite(Behavior):
    def forward(self, ng: NeuronGroup):
        for synapse in ng.afferent_synapses["All"]:
            ng.I += synapse.I
        ng.I[ng.I<0]=0