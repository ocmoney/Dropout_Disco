from pathlib import Path

class Tokeniser:
    
    def __init__(self):
        path = Path('./CBOWs/text8').absolute()
        self.identifyers = {}
    
        with open(path, 'r') as f:
            f = f.read()
            f = f.split(' ')

        for i in range(len(f)):

            if f[i] not in self.identifyers:

                self.identifyers[f[i]] = len(self.identifyers) + 1

    def getCorpusSize(self) -> int:
        return len(self.identifyers)

    def getIdentifyer(self, token) -> int:
        return self.identifyers.get(token, "no identifyer")