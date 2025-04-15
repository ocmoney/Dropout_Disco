from tokeniser import Tokeniser
from model import CROW as Model
import torch 

tokeniser = Tokeniser()
identifier = tokeniser.getIdentifyer("Hello my is Bes")
tokens = torch.tensor(identifier)

# print(tokeniser.getCorpusSize())

model = Model(tokeniser.getCorpusSize())
res = model.forward(tokens)
print(res.shape)
print(res)