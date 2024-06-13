import numpy as np
from collections import defaultdict

class Tracker:
    def __init__(self, ics, pdes, datas):
        self.ics = ics
        self.pdes = pdes
        self.datas = datas
        
        self.num_terms = len(self.ics) + len(self.pdes) + len(self.datas)
        self.reset()

    def update(self, total_loss, losses):
        if not len(losses) == self.num_terms:
            raise ValueError("length of loss is not correct.")
        
        self.log['total'].append(total_loss)
        
        idx = 0
        for ic in range(len(self.ics)):
            self.log[f'I_{self.ics[ic]}'].append(losses[idx])
            idx += 1
            
        for s in range(len(self.pdes)):
            self.log[f'P_{self.pdes[s]}'].append(losses[idx])
            idx += 1
        
        for m in range(len(self.datas)):
            self.log[f'D_{self.datas[m]}'].append(losses[idx])
            idx += 1
            
    def reset(self):
        self.log = defaultdict(list)
        self.history = dict()

    def get_history(self):
        for k, v in self.log.items():
            self.history[k] = np.mean(v)
            
        return self.history