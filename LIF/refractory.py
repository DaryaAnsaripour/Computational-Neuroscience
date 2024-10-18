from pymonntorch import *

class Refractory(Behavior):
    def initialize(self, ng):
        ng.I[ng.u > -70].fill_(0)

    def forward(self, ng):
        ng.I[ng.u>-70].fill_(0)
        