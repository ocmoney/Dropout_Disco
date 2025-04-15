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
        # Ensure input is long tensor
        if inputs.dtype != torch.long:
            inputs = inputs.long()
        
        # inputs shape: [batch_size, context_size*2]
        embs = self.emb(inputs)       # shape: [batch_size, context_size*2, embedding_dim]
        
        # Average over context window dimension
        embs = embs.mean(dim=1)       # shape: [batch_size, embedding_dim]
        
        # Linear layer
        out = self.linear(embs)       # shape: [batch_size, vocab_size]
        
        # Apply log softmax over vocabulary dimension
        probs = F.log_softmax(out, dim=1)  # shape: [batch_size, vocab_size]
        
        return probs

