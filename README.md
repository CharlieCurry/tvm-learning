# TVM-Learning

- *autotvm-------------主要介绍 tvm.autotvm中 tuner的使用(针对给定的schedule调参)*
- *learningfromlog-----利用机器学习方法对已获得的 log数据进行分析*
- *schedule stage------tvm.schedule中定义的可选择的 stage(优化技术的使用)*
- *scheduling----------tvm不能自动选择合适的 schedule,需要手动编写合适的 schedule,不同的 schedule在还未选择参数时就有很大的差异*
- *autoschedule--------不同的schedule定义代表着不同的变换，同时也决定了后续参数调优的空间，那么能够自动选择schedule么*
- *end2end-------------对已有模型进行端到端的优化*
- *NNTuner-------------基于神经网络的Tuner，尝试替换xgboost tuner*
- *2thconferenceslide--第二次会议报告*
## Open source stack
repo | src
---|---
TVM官网|https://tvm.ai/
tvm Benchmark |https://github.com/apache/incubator-tvm/wiki/Benchmark#mobile-gpu
RELEASE |https://bitbucket.org/act-lab/release/src/master/
chameleon |https://bitbucket.org/act-lab/chameleon/src/master/
d2l-tvm |https://github.com/d2l-ai/d2l-tvm
tvm-cuda-int8-benchmark| https://github.com/vinx13/tvm-cuda-int8-benchmark
tvm-distro| https://github.com/uwsampl/tvm-distro
知乎专栏:深度学习编译器学习笔记和实践体会| https://zhuanlan.zhihu.com/c_1169609848697663488
###### This tutorial focuses on the end-to-end optimization of TVM for deep learning model, especially for the innovation of autotvm module.Organized as Three Parts:
## Part I Easy to use TVM
### 安装要求

```
gcc版本 >=4.8
CMake >=3.5
python3 最新的tvm已经不支持python2了
llvm 我选择的版本为llvm-4.0.0
```
### 从github上下载TVM

```
git clone --recursive https://github.com/dmlc/tvm

git clone --recursive https://github.com/CharlieCurry/incubator-tvm.git

sudo apt-get update

sudo apt-get install -y python python-dev python-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake

```
### 创建build文件，并拷贝修改config配置文件
###### 在tvm目录下创建build文件，并将tvm/cmake下的config.camke文件拷贝到build目录下

```
	cd tvm
	mkdir build
	cp cmake/config.cmake build
```
###### 打开build下的config.cmake文件，因为我需要支持CUDA与llvm环境，所以找到下面的配置并设置ON，如果要使用cudnn，就去打开cudnn的开关，根据自己的需求来

```
set(USE_CUDA OFF)     --->set(USE_CUDA ON)  
set(USE_LLVM OFF)     --->set(USE_LLVM ON)
```
### 编译
###### 修改好配置文件后，进行编译。因为修改了两个编译选项，因此首先需要cmake重新生成Makefile，以后每次新添加了文件和文件夹，一定要重新cmake，否则文件很可能没有编译。

```
  	cd build
  	cmake -DCMAKE_BUILD_TYPE=Debug ..   //如果需要gdb跟踪源码的话需要加-DCMAKE_BUILD_TYPE=Debug
   	make -j4
```
### 添加环境变量

```
vim ~/.bashrc
添加：
export TVM_PATH=/chy/tvm
export PYTHONPATH=$TVM_PATH/python:$TVM_PATH/topi/python:$TVM_PATH/nnvm/python:${PYTHONPATH}
source ~/.bashrc
```
### 可能出现的问题
1.fatal error:cmake没出错，但是make的时候出错，（动态链接出错）

```
（1）fatal error: dlpack/dlpack.h: No such file or directory
（2）fatal error: dmlc/logging.h: No such file or directory
（3）fatal error: dmlc/json.h: No such file or directory
```


```
root@chi:/chy# git clone --recursive https://github.com/CharlieCurry/incubator-tvm.git tvm
Cloning into 'tvm'...
remote: Enumerating objects: 56532, done.
remote: Total 56532 (delta 0), reused 0 (delta 0), pack-reused 56532
Receiving objects: 100% (56532/56532), 20.24 MiB | 154.00 KiB/s, done.
Resolving deltas: 100% (38925/38925), done.
Checking out files: 100% (2197/2197), done.
Submodule 'dlpack' (https://github.com/dmlc/dlpack) registered for path '3rdparty/dlpack'
Submodule 'dmlc-core' (https://github.com/dmlc/dmlc-core) registered for path '3rdparty/dmlc-core'
Submodule '3rdparty/rang' (https://github.com/agauniyal/rang) registered for path '3rdparty/rang'
Cloning into '/chy/tvm/3rdparty/dlpack'...
remote: Enumerating objects: 13, done.
remote: Counting objects: 100% (13/13), done.
remote: Compressing objects: 100% (8/8), done.
remote: Total 175 (delta 3), reused 2 (delta 1), pack-reused 162
Receiving objects: 100% (175/175), 62.00 KiB | 262.00 KiB/s, done.
Resolving deltas: 100% (59/59), done.
Cloning into '/chy/tvm/3rdparty/dmlc-core'...
remote: Enumerating objects: 18, done.
remote: Counting objects: 100% (18/18), done.
remote: Compressing objects: 100% (17/17), done.
remote: Total 5978 (delta 2), reused 5 (delta 0), pack-reused 5960
Receiving objects: 100% (5978/5978), 1.54 MiB | 725.00 KiB/s, done.
Resolving deltas: 100% (3625/3625), done.
Cloning into '/chy/tvm/3rdparty/rang'...
remote: Enumerating objects: 704, done.
remote: Total 704 (delta 0), reused 0 (delta 0), pack-reused 704
Receiving objects: 100% (704/704), 256.13 KiB | 12.00 KiB/s, done.
Resolving deltas: 100% (362/362), done.
Submodule path '3rdparty/dlpack': checked out '0acb731e0e43d15deee27b66f10e4c5b4e667913'
Submodule path '3rdparty/dmlc-core': checked out '808f485387f9a03f78fa9f1159f387d0d91b7a28'
Submodule path '3rdparty/rang': checked out 'cabe04d6d6b05356fa8f9741704924788f0dd762'
```

## Part II  Latest development


## Part III  Contribution

###### Contact me:chihaoyu@cigit.ac.cn
