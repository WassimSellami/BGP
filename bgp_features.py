import time

class BGPFeatures:
    def __init__(self):
        self.nb_A = 0
        self.nb_W = 0
        self.last_reset = time.time()
    
    def reset(self):
        self.nb_A = 0
        self.nb_W = 0
        self.last_reset = time.time()
    
    @property
    def nb_A_W(self):
        return self.nb_A + self.nb_W

    def classify_elem(self, elementType):
        if elementType == 'A':
            self.nb_A += 1
        elif elementType == 'W':
            self.nb_W += 1


