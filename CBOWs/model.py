import torch 
import torch.nn as nn
import torch.nn.functional as F

### Hyperparameters Start ###
EMBEDDING_DIM = 9
### Hyperparameters End ###
        
class CROW(torch.nn.Module):    
    def __init__(self, numOfEmbedings):
        super().__init__()
        self.emb = torch.nn.Embedding(numOfEmbedings, EMBEDDING_DIM)
        self.linear = torch.nn.Linear(EMBEDDING_DIM, numOfEmbedings)
        
    def forward(self, inputs):
        
        # print(inputs)
        embs = self.emb(inputs)       
        
        embs = embs.mean(dim=0, keepdim=True)
        
        out = self.linear(embs)
        
        probs = F.log_softmax(out, dim=1)
        
        return probs
        