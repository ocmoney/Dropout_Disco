from pathlib import Path

import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = Path('./CBOWs/text8').absolute()

with open(path, 'r') as f:
    
    f = f.read()
    
    f = f.split(' ')
    

identifyer = {}

for i in range(len(f)):
    
    if f[i] not in identifyer:
                
        identifyer[f[i]] = len(identifyer) + 1


print(identifyer)


class CBOW(nn.Module):
    
    def __init__(self):
        super(CBOW, self).__init__()
