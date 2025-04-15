from pathlib import Path
import torch

class Tokeniser:
    
    def __init__(self):
        path = Path('./CBOWs/text8').absolute()
        self.identifyers = {}
    
        with open(path, 'r') as f:
            f = f.read()
            f = f.lower()
            f = f.split(' ')

        for i in range(len(f)):

            if f[i] not in self.identifyers:

                self.identifyers[f[i]] = len(self.identifyers)

    def getCorpusSize(self) -> int:
        return len(self.identifyers)

    def getIdentifyer(self, tokens: str) -> list[int]:
        tokens = tokens.lower()
        tokens = tokens.split(" ")

        # Return 0 for unknown tokens instead of "no identifyer"
        return [self.identifyers.get(token, 0) for token in tokens]
    
    def get_predicted_word(self, probs):
        # Get the word index with highest probability
        predicted_idx = torch.argmax(probs, dim=1)
                
        reverse_dict = {v: k for k, v in self.identifyers.items()}
        predicted_word = reverse_dict.get(predicted_idx.item(), "<UNK>")
        
        return predicted_word