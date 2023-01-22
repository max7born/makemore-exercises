import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

NUM_CHARS = 27

class Dataset:

  def __init__(self, path: str) -> None:
    self.words = open(path, 'r').read().splitlines()
    chars = sorted(list(set(''.join(self.words))))
    self.ctoi = {c: i+1 for i, c in enumerate(chars)}
    self.ctoi['.'] = 0
    self.itoc = {i: c for c, i in self.ctoi.items()}


class NGramExplicitModel:

  def __init__(self, n: int, data: Dataset) -> None:
    self.n = n
    self.data = data

  def count(self):
    self.counts = torch.zeros(tuple([NUM_CHARS for _ in range(self.n)]), dtype=torch.int32)
    for w in self.data.words:
      context = [0] * (self.n-1)
      for c in w + '.':
        indices = tuple(context + [self.data.ctoi[c]])
        self.counts[indices] += 1
        context = context[1:] + [self.data.ctoi[c]]
      
    self.P = F.normalize((self.counts+1).float(), p=1, dim=-1)
  
  def sample(self, generator: torch.Generator, num_samples: int = 1):
    for i in range(num_samples):
      sample = []
      context = [0 for _ in range(self.n-1)]
      while True:
        ix = torch.multinomial(self.P[tuple(context)], num_samples=1, replacement=True, generator=generator).item()
        sample.append(self.data.itoc[ix])
        if ix == 0:
          break
        context = context[1:] + [ix]

      print(''.join(sample))



class NGramMLPModel:

  def __init__(self, n: int, data: Dataset) -> None:
    self.n = n
    self.data = data



def main():
  g = torch.Generator().manual_seed(2147483647)
  data = Dataset('data/names.txt')
  model = NGramExplicitModel(n=4, data=data)
  model.count()
  model.sample(generator=g, num_samples=5)

  


if __name__=='__main__':
  main()


