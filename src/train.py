#load data train 
#load data test 

#trainning loop 
#save weights  
from mlp_model import MLPModel as model

m = model()

token =  m.tokenise("test out this" )

out = m.forward(token)

print(out)
#testing loop