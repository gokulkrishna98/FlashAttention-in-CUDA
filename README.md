# FlashAttention in CUDA

> Note: This is just a experimental implementation to compare performance.
> Use it with skepticism

# Introduction
In this project, we will implement baseline gpu code and then flash-attention gpu code, and check performance.

# Build Steps
- Install CUDA in the device you want to run this.
- Download libtorch for c++ and unzip the libtorch in the project root folder. You could
either download it from nightly builds:
```console
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
```
or from pytorch download page
```console
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcpu.zip
```
- Now you create build folder and build using cmake:
```console
mkdir build
cd build
cmake ..
make -j32
```
- run the executable
```console
./fa
```

# References
- Optimizing Matrix Multiplication - [here](https://siboehm.com/articles/22/CUDA-MMM)
- Flash Attention by Tri Dao - [here](https://arxiv.org/abs/2205.14135)
- Minimal flash attention inspiration - [here](https://github.com/tspeterkim/flash-attention-minimal)

> TODO:
> - Add command line arguments to executable.
> - Add improve the flash attention kernel support to good amount of dimension 
> - Write a article and link it here.
