from pathlib import Path

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