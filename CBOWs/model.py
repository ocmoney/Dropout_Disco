import torch 
import torch.nn as nn
import torch.nn.functional as F

class CBOW(nn.Module):
    
    def __init__(self, corpusSize:int):
        
        super().__init__()
        
        self.emb = nn.EmbeddingBag(corpusSize, 100)        
        
    def forward(self, x):#
        print(x)
        # return self.emb(x)