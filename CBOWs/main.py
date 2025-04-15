from tokeniser import Tokeniser
from model import CROW as Model
import torch 

def loss(probs):
    probs = torch.exp(probs)
    entropy = -torch.sum(probs * probs.log())
    return entropy.item()


tokeniser = Tokeniser()
identifier = tokeniser.getIdentifyer("Hello my is Bes")
tokens = torch.tensor(identifier)

model = Model(tokeniser.getCorpusSize())
res = model.forward(tokens)

# print(tokeniser.get_predicted_word(res))

print(loss(res))