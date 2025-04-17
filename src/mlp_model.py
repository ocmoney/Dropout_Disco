import torch
import torch.nn as nn 

from word2vec import Vocabulary

class MLPModel(nn.Module):
    def __init__(self):
        
        super().__init__() 
        
        vocab_path = "../models/word2vec/text8_vocab_NWAll_MF5.json"        
        self.vocab = Vocabulary.load_vocab(vocab_path)
    
        vocab_model = "../models/word2vec/CBOW_D128_W5_NWAll_MF5_E15_LR0.001_BS512/model_state.pth"
        cbow = torch.load(vocab_model, map_location=torch.device("cpu"))
    
        self.emb = nn.Embedding.from_pretrained(cbow['embeddings.weight'])

        self.l1 = nn.Linear(128, 96)
        self.r1 = nn.ReLU()
        self.b1 = nn.BatchNorm1d(96)
        self.d1 = nn.Dropout(.2)
    
        self.l2 = nn.Linear(96, 64)
        self.r2 = nn.ReLU()
        self.b2 = nn.BatchNorm1d(64)
        self.d2 = nn.Dropout(.2)
        
        self.l3 = nn.Linear(64, 32)
        self.r3 = nn.ReLU()
        self.b3 = nn.BatchNorm1d(32)
        self.d3 = nn.Dropout(.2)
        
        self.l4 = nn.Linear(32, 16)
        self.r4 = nn.ReLU()
        self.b4 = nn.BatchNorm1d(16)
        self.d4 = nn.Dropout(.2)
        
        self.out = nn.Linear(16, 1)
    
    def tokenise(self, text):
        
        tokens = text.lower().split()
        indices = [self.vocab.get_index(token) for token in tokens]   
        return torch.tensor(indices, dtype=torch.long)
    
    def forward(self, input):
        
        l0 = self.emb(input)
        
        l1 = self.l1(l0)
        r1 = self.r1(l1)
        b1 = self.b1(r1)
        d1 = self.d1(b1)

        l2 = self.l2(d1)
        r2 = self.r2(l2)
        b2 = self.b2(r2)
        d2 = self.d2(b2)
        
        l3 = self.l3(d2)
        r3 = self.r3(l3)
        b3 = self.b3(r3)
        d3 = self.d3(b3)
        
        l4 = self.l4(d3)
        r4 = self.r4(l4)
        b4 = self.b4(r4)
        d4 = self.d4(b4)
        
        out = self.out(b4)
        
        return out
        
        


