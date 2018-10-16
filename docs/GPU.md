# GPU

## What is a GPU?

GPUs are specialized hardware originally created to render games in high frame rates.  
Graphics texturing and shading requires highly parallel matrix and vector operations.

It so happens that Deep Learning also requires super fast matrix computations.  
If you would like to train anything meaningful in deep learning, a GPU is highly recommended - specifically an Nvidia GPU.

<br>
## How to get access to one

Generally 3 options to choose from...
1. Jupyter Notebook with GPU enabled
    * [Sagemaker]()
    * [Gradient]()
    * [Salamandar]()
2. Cloud Server with access to GPU 
    * [GCP]()
    * [AWS EC2]()
    * [Paperspace]()
3. Personal Computer with GPU hardware installed

**For those starting out, we highly recommend Jupyter Notebooks (Option 1)**
* Notebooks are the easiest way to start writing python code and running fastai.  
* Renting a Cloud Server (Option 2) requires environment configuration and setup.  
* Building a PC (Option 3) requires environment setup and more up-front money. Obviously a moot point if you already own a gaming PC.  

For [Part 2](http://course.fast.ai/part2.html) of the course, we will go into more specific details and benefits on both building a PC or running a server.  

<br>
## GPU vs CPU

| GPU                     | CPU                      |
| ----------------------- |:------------------------:|
| Optimized FP Operations | Complex Instruction Set  |
| Slow (1-2 Ghz)          | Fast (3-4 Ghz)           |
| > 1000 Cores            | < 100 Cores              |
| Fast Dedicated VRAM     | Large Capacity System RAM|

Deep Learning really only cares about the number of Floating Point Operations (FLOPs) per second. GPUs are highly optimized for that. 


![gpu_cpu_comparison.png](images/gpu_cpu_comparison.png)

In the chart above, you can see that GPUs (red/green) can theortically do 10-15x the operations of CPUs (in blue).  

This speedup very much applies in practice too. 

**But don't just take our word for it!**  
Try running this inside a Jupyter Notebook:

Cell [1]:  
`import torch`  
`t_cpu = torch.rand(500,500,500)`  
`%timeit t_cpu @ t_cpu`  

Cell [2]:  
`t_gpu = torch.rand(500,500,500).cuda()`  
`%timeit t_gpu @ t_gpu`  

<br>
## Use Nvidia
1. Currently the fastest
2. Native Pytorch support for CUDA
3. Highly optimized for deep learning with cuDNN