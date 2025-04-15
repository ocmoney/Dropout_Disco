from tokeniser import Tokeniser
from model import CBOW as Model


tokeniser = Tokeniser()

model = Model(tokeniser.getCorpusSize())

token = tokeniser.getIdentifyer("the")

# print(token)

res = model.forward(token)

print(res)