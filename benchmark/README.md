
## M3 Pro:

```sh
(base)  benchmark  master @ python mps_benchmark.py
CPU Time: 1.4883 seconds
MPS Time: 0.0002 seconds
(base)  benchmark  master @ python mps_benchmark.py
CPU Time: 1.4697 seconds
MPS Time: 0.0001 seconds
(base)  benchmark  master @ python mps_benchmark.py
CPU Time: 1.4773 seconds
MPS Time: 0.0002 seconds
(base)  benchmark  master @
```

## 1080 Cuda:

```sh
(base) xlisp@xlisp:~/visualization-machine-learning/benchmark$ python cuda_benchmark.py
CPU Time: 4.8574 seconds
CUDA Time: 0.5576 seconds
(base) xlisp@xlisp:~/visualization-machine-learning/benchmark$ python cuda_benchmark.py
CPU Time: 4.9740 seconds
CUDA Time: 0.5570 seconds
(base) xlisp@xlisp:~/visualization-machine-learning/benchmark$ python cuda_benchmark.py
CPU Time: 5.0024 seconds
CUDA Time: 0.5586 seconds
(base) xlisp@xlisp:~/visualization-machine-learning/benchmark$
```
