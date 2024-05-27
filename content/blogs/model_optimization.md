---
title: "Model Optimization"
date: 2024-01-12T14:34:17+05:45
draft: false

cover:
    image: 'img/optimization.png'
    alt: 'Model Optimization'
    caption: 'Model Optimization on Deep Learning Models'
    relative: false
    hidden: false
tags: ["ai", "ml"]
categories: ["tech", "optimization"]
---


# Model Optimization

Most deep learning models are made up of millions/billions of parameters. So, when we inference a model, we need to load all its parameter in the memory, this means big models cannot be loaded easily on a edge/embedded devices. So, this blog is focused on optimizing a model through different techniques and analyze its effect on inference time, model size, accuracy. 

Model optimization is transforming machine/deep learning model such that it has:

1. Faster Inference Time

2. Reduction in network size (memory efficient)

We're going to cover different topics, so buckle up

1. Quantization:  
Reducing number of bits used for model parameters

2. Network Pruning:  
Reducing the number of distinct weight values

3. Low-Rank Matrix Factorization:  
Factorizing model weights such that it reduces overall model parameters

4. Knoweldge Distillation:  
Obtatining a smaller netwokr by mimicking the prediction


![Optimization](https://raw.githubusercontent.com/shulavkarki/shulavkarki.github.io/master/static/img/model_optimization/types_diagram.png)

We'll mainly focus on optimizing pytorch model. But the gist we'll be same for Tensorflow/caffe.,etc


## 1. Quantization  
Quantization means converting/reducing the model parameters precision to lower bit precision typically 8-bit integers , without retraining the model. 


### Remapping from float32 to int8:  
![Quantization](https://raw.githubusercontent.com/shulavkarki/shulavkarki.github.io/master/static/img/model_optimization/quantization.png)

### Pytorch Availability  

#### 1. Half Precision  
Half precision refers to performing computations using 16-bit floating point numbers(half precision) instead of standard 32-bit floating point numbers(single precision) to accelerate inference process.

Using half precision, the memory requirements and computational cost of the inference process can be reduced, leading to faster inference time.

Blockers:  
i. As of now, Half-Precision is not [supported](https://stackoverflow.com/questions/62112534/fp16-inference-on-cpu-pytorch) for CPU. Operations for conv_layer, stft, etc doesn't support operations in float-16 on cpu.

#### 2. [Post Training Dyanmic Quantization](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html)  
In Dynamic Quantization, the quantization on model weights are quantized ahead of time while quantization on activatios occurs dynamically during runtime.  
During inference, the activation's are collected for input and are analyzed to determine their dynamic range. Once the dynamic range is known, quantization parameters such as scale and zero-point are calculated to map the floating point activation to lower precision.

```python
import torch
model = …  #pytorch model
quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
output = quantized_model(input) #infer on quantized model

```

#### - [Supported Layers](https://pytorch.org/docs/stable/quantization.html#torch.quantization.quantize_dynamic):  
1. nn.Linear
2. nn.LSTM / nn.GRU / nn.RNN
3. nn.Embedding


#### 3. Post Trianing Static Quantization  
Scales and zero points for all the activation tensors are pre-computed using some representative unlabeled data.

Given a floating point DNN, we would just have to run the DNN using some representative unlabeled data, collect the distribution statistics(scale point and zero point) for all the activation layers.