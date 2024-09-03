In [4]:
   ...: import torch
   ...: import torch.nn as nn
   ...: import torch.optim as optim
   ...: import torch.nn.functional as F
   ...: from torchvision import datasets, transforms
   ...: from torch.utils.data import DataLoader



In [6]: class Net(nn.Module):
   ...:     def __init__(self):
   ...:         super(Net, self).__init__()
   ...:         self.fc1 = nn.Linear(28 * 28, 128)
   ...:         self.fc2 = nn.Linear(128, 64)
   ...:         self.fc3 = nn.Linear(64, 10)
   ...:
   ...:     def forward(self, x):
   ...:         x = x.view(-1, 28 * 28)
   ...:         x = F.relu(self.fc1(x))
   ...:         x = F.relu(self.fc2(x))
   ...:         x = self.fc3(x)
   ...:         return x
   ...:


In [8]: model = Net()
   ...:

In [9]: model.load_state_dict(torch.load("mnist_model.pth"))
<ipython-input-9-6e8b1d6c4e1d>:1: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load("mnist_model.pth"))
Out[9]: <All keys matched successfully>

In [10]: model.eval()
    ...:
Out[10]:
Net(
  (fc1): Linear(in_features=784, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=10, bias=True)
)

In [11]: dummy_input = torch.randn(1, 1, 28, 28)
    ...:

In [12]: dummy_input
Out[12]:
tensor([[[[ 1.6684e+00, -4.3599e-01,  1.3496e+00, -2.3099e-02, -6.3655e-01,
           -2.4368e+00,  8.6224e-01,  2.9872e-01],
 .....
          [-3.4220e-01,  7.5400e-01, -2.4024e-01,  4.1501e-01, -9.4485e-01,
           -7.5373e-01,  1.6971e+00,  3.6350e-02,  9.7232e-02,  6.7685e-02,
           -2.1645e-02,  1.1723e+00, -1.1964e+00,  3.2167e-01,  6.9059e-01,
           -9.4678e-02,  2.6423e+00, -2.7849e-01,  2.6289e-01, -6.4902e-02,
            7.2097e-01,  1.0745e+00,  1.9120e+00,  2.7027e+00, -2.6037e+00,
           -4.2325e-03,  3.5967e-01, -1.1734e+00]]]])

In [13]: dummy_input.shape
Out[13]: torch.Size([1, 1, 28, 28])

##  pip install onnx

In [15]: torch.onnx.export(model, dummy_input, "mnist_model.onnx")

In [16]:

