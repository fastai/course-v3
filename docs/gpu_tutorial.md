---

title: GPU
sidebar: home_sidebar

---
## What is a GPU?

GPUs are specialized hardware originally created to render games in high frame rates. Graphics texturing and shading require a lot of matrix and vector operations executed in parallel and those chips have been created to take the heat off the CPU while doing that. 

## Why a GPU?

It so happens that Deep Learning also requires super fast matrix computations. So researchers put two and two together and [started training models in GPU's](http://www.machinelearning.org/archive/icml2009/papers/218.pdf) and the rest is history.

| GPU                     |            CPU            |
| ----------------------- | :-----------------------: |
| Optimized FP Operations |  Complex Instruction Set  |
| Slow (1-2 Ghz)          |      Fast (3-4 Ghz)       |
| > 1000 Cores            |        < 100 Cores        |
| Fast Dedicated VRAM     | Large Capacity System RAM |

Deep Learning really only cares about the number of Floating Point Operations (FLOPs) per second. GPUs are highly optimized for that. 

<img alt="gpu_cpu_comparison" src="/images/gpu/gpu_cpu_comparison.png" class="screenshot">

In the chart above, you can see that GPUs (red/green) can theoretically do 10-15x the operations of CPUs (in blue).  This speedup very much applies in practice too. **But do not take our word for it!**  

Try running this inside a Jupyter Notebook:

Cell [1]:  
`import torch`  
`t_cpu = torch.rand(500,500,500)`  
`%timeit t_cpu @ t_cpu`  

Cell [2]:  
`t_gpu = torch.rand(500,500,500).cuda()`  
`%timeit t_gpu @ t_gpu`  

If you would like to train anything meaningful in deep learning, a GPU is what you need - specifically an NVIDIA GPU.

<br>

## Why NVIDIA?

We recommend you to use an NVIDIA GPU since they are currently the best out there for a few reasons:

1. Currently the fastest

2. Native Pytorch support for CUDA

3. Highly optimized for deep learning with cuDNN

---

*Many thanks to Andrew Shaw for writing this guide.*
