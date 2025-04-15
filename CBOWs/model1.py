import torch
import torch.nn as nn
import torch.nn.functional as F

vocab = {
    "Hello": 72,
    "my": 44,
    "name": 21,
    "is": 93,
    "Bes": 11
}

sentence = [vocab[word] for word in ["Hello", "my", "is", "Bes"]]

print(sentence)

class CROW(torch.nn.Module):    
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(128, 9)
        self.linear = torch.nn.Linear(9, 128)
        
    def forward(self, inputs):
    
        embs = self.emb(inputs)
        # print(embs.shape)
        
        
        # [seq_len, embed_dim] -> [1, embed_dim]
        embs = embs.mean(dim=0, keepdim=True)
        # print(embs)
        
        # [1, embed_dim] -> [1, vocab_size]
        out = self.linear(embs)
        # print(out)
        
        probs = F.log_softmax(out, dim=1)
        # print(probs)
        
        return probs
    
model = CROW()
input_tensor = torch.tensor(sentence)
res = model.forward(input_tensor)

# print(res)