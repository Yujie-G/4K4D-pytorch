# Learning NeRF

This repository is initially created by [Haotong Lin](https://haotongl.github.io/).
Forked from [LearningNeRF](https://github.com/pengsida/learning_nerf)

## TODO
- [x] Add a simple synthetic dataset
- [x] Add a simple NeRF network
- [ ] Add importance sampling method
- [ ] Adapt to other dataset

## Result

without importance sampling, 300 epochs, PSNR: 24.914
![img.png](assets/imgs/img.png)


## Introduction
### 配置文件

我们已经在configs/nerf/ 创建好了一个配置文件，nerf.yaml。其中包含了复现NeRF必要的参数。
你可以根据自己的喜好调整对应的参数的名称和风格。


### 创建dataset： lib.datasets.nerf.synthetic.py

核心函数包括：init, getitem, len.

init函数负责从磁盘中load指定格式的文件，计算并存储为特定形式。

getitem函数负责在运行时提供给网络一次训练需要的输入，以及groundtruth的输出。
例如对NeRF，分别是1024条rays以及1024个RGB值。

len函数是训练或者测试的数量。getitem函数获得的index值通常是[0, len-1]。


#### debug dataset：

```
python run.py --type dataset --cfg_file configs/nerf/nerf.yaml
```

### 创建network:

核心函数包括：init, forward.

init函数负责定义网络所必需的模块，forward函数负责接收dataset的输出，利用定义好的模块，计算输出。例如，对于NeRF来说，我们需要在init中定义两个mlp以及encoding方式，在forward函数中，使用rays完成计算。


#### debug：

```
python run.py --type network --cfg_file configs/nerf/nerf.yaml
```

### loss模块和evaluator模块

这两个模块较为简单，不作仔细描述。

debug方式分别为：

```
python train_net.py --cfg_file configs/nerf/nerf.yaml
```

```
python run.py --type evaluate --cfg_file configs/nerf/nerf.yaml
```
