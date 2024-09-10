import torch
import torch.nn as nn

#### https://stackoverflow.com/questions/75542366/pytorch-what-module-should-i-use-to-multiply-the-output-of-a-layer-using-seque
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-custom-nn-modules

class Multiply(nn.Module):
    def __init__(self, N,M):
        super().__init__()
        self.N = N
        self.M = M
        self.weight = torch.nn.Parameter(torch.rand(self.N,self.M))

    def forward(self, x):
        x = torch.multiply(self.weight,x)
        return x


if __name__ == '__main__':
   mul = Multiply(3)
   x = torch.ones(3)
   y = mul(x)

   criterion = nn.MSELoss(reduction='sum')
   optimizer = torch.optim.SGD(mul.parameters(), lr=0.01)
   y = torch.sin(x)

   for t in range(200):
       # Forward pass: Compute predicted y by passing x to the model
       y_pred = mul(x)

       # Compute and print loss
       loss = criterion(y_pred, y)
       print(t, loss.item())

       # Zero gradients, perform a backward pass, and update the weights.
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()


   qq = 0
