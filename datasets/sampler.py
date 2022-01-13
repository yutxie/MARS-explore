import random


class StreamSampler():
    def __init__(self, S=5000):
        self.N = 0
        self.S = S
        self.data = []

    def update(self, data):
        for item in data:
            self.N += 1
            if len(self.data) < self.S:
                self.data.append(item)
            elif random.random() < 1. * self.S / self.N:
                i = random.randint(0, self.S-1)
                self.data[i] = item
            else: continue

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.data[index]