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
        ix = self.data.ctoi[c]
        indices = tuple(context + [ix])
        self.counts[indices] += 1
        context = context[1:] + [ix]
      
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

  def __init__(self, n: int, data: Dataset, embed_dim: int) -> None:
    self.n = n
    self.data = data
    self.embed_dim = embed_dim
    self.prepare_ds()
    self.init_network()

  def prepare_ds(self):
    # Create train set
    self.xs, self.ys = [], []
    for w in self.data.words:
      context = [0] * (self.n-1) 
      for c in w + '.':
        ix = self.data.ctoi[c]
        self.xs.append(context)
        self.ys.append(self.data.ctoi[c])
        context = context[1:] + [ix]
    self.xs = torch.tensor(self.xs)
    self.ys = torch.tensor(self.ys)

  def init_network(self):
    g = torch.Generator().manual_seed(2147483647)
    self.C = torch.randn((NUM_CHARS, self.embed_dim), generator=g, requires_grad=True)
    # C[xs].shape = (num_data_pairs, context_len, embed_dim)
    self.W = torch.randn(((self.n-1)*self.embed_dim, NUM_CHARS), generator=g, requires_grad=True)
    self.params = [self.C, self.W]
    for p in self.params:
      p.requires_grad = True


  def train_network(self, num_iters: int, lr: float):
    for k in range(num_iters):
      # NN forward pass
      xenc = self.C[self.xs]
      xenc = xenc.view(-1, (self.n-1)*self.embed_dim)
      logits = xenc @ self.W             # log counts -> only thing that will change in Transformers
      counts = logits.exp()              # equivalent to counts
      P = counts / counts.sum(dim=1, keepdims=True)
      # last 2 lines: softmax
      
      # loss: negative llh of probs corresponding to true labels
      loss = -P[torch.arange(self.ys.nelement()), self.ys].log().mean() + 0.01*(self.W**2).mean()
      
      ## NN backward pass
      self.W.grad = None       # set grad to 0
      loss.backward()
      if k%10 == 0:
        print(f'Iter {k}, loss {loss.item()}')

      self.W.data += -lr*self.W.grad


  def sample(self, generator: torch.Generator, num_samples: int):
    for i in range(num_samples):
      sample = []
      context = [0 for _ in range(self.n-1)]
      while True:
        xenc = self.C[torch.tensor(context)].flatten().unsqueeze(0)
        logits = xenc @ self.W
        counts = logits.exp()
        p = F.normalize(counts.float(), p=1, dim=-1)

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=generator).item()
        sample.append(self.data.itoc[ix])
        if ix == 0:
          break
        context = context[1:] + [ix]

      print(''.join(sample))



def main():
  g = torch.Generator().manual_seed(2147483647)
  data = Dataset('data/names.txt')
  # model = NGramExplicitModel(n=4, data=data)
  model = NGramMLPModel(2, data, 10)
  model.train_network(250, 1)
  model.sample(generator=g, num_samples=5)

  
if __name__=='__main__':
  main()


