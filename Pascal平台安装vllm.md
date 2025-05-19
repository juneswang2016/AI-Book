## Pascal平台(SM60)安装vllm 0.85

# 构建编译环境
1、安装推荐编译器
 ```
  sudo apt-get update  -y
  sudo apt-get install -y gcc-12 g++-12 libnuma-dev
 ```
2、安装 vLLM 构建所需 Python 包
```
pip install --upgrade pip
pip install "cmake>=3.26" wheel packaging ninja "setuptools-scm>=8" numpy
```
3、安装cuda环境，<span class="red-text">并跳过驱动安装</span>
```
wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda_12.9.0_575.51.03_linux.run
sudo sh cuda_12.9.0_575.51.03_linux.run
```
配置环境变量‌
在 ~/.bashrc 或 ~/.zshrc 中添加：

```
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
执行 source ~/.bashrc 使配置生效。

# 编译

首先你需要有一份vllm 0.8.5的项目代码
```
git clone -b v0.8.5 --single-branch https://github.com/vllm-project/vllm.git
```
下载后：
1、打开项目根目录下的CMakeLists.txt，找到34行的CUDA_SUPPORTED_ARCHS，并加上6.0;6.1

最后看起来应该像下面这样：
set(CUDA_SUPPORTED_ARCHS "6.0;6.1;7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0")

2、因为vllm的cuda代码用了half精度的atomicAdd，然而pascal架构的计算能力并不支持这个操作
找到项目目录下csrc/moe/moe_wna16.cu，添加以下代码：
```
#ifdef __CUDA_ARCH__

#if __CUDA_ARCH__ < 700
__device__ __forceinline__ void atomicAdd(half* address, half val) {
    unsigned int * address_as_ui =
      (unsigned int *) ((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

  do {
    assumed = old;
    __half_raw hsum;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    half tmpres = __hadd(hsum, val);
    hsum = __half_raw(tmpres);
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
   } while (assumed != old);
}
#endif
#endif
```
3、降低block size或者num stages
解决triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 65536, Hardware limit: 49152. Reducing block sizes or `num_stages` may help.错误
找到python文件夹下site-packages/vllm/attention/ops/prefix_prefill.py 903行，把BLOCK_M BLOCK_N各降低一半

不出意外pip install -e .可以正确编译安装。<span class="red-text">（需要开启梯子，编译时要连接github）</span>

# 运行
1、如果遇到pascal gpu被pytorch告知不被支持错误
https://github.com/sasha0552/pascal-pkgs-ci
解决办法：终端运行
```
sed -e "s/.major < 7/.major < 6/g"                                 \

    -e "s/.major >= 7/.major >= 6/g"                               \

    -i                                                             \

    venv/lib/python3.10/site-packages/torch/_inductor/scheduler.py \

    venv/lib/python3.10/site-packages/torch/utils/_triton.py
```
记得把路径改成你的python环境和版本

2、继续运行，然后你的pascal设备会被triton判死刑

解决办法：删除triton，下载专门调教成适配pascal架构的triton3.2.0版本，链接：

https://sasha0552.github.io/pascal-pkgs-ci/

不建议下载此站预编译好的vllm-pascal，可能会出现神秘问题

注：一定要在删除triton后再安装triton-pascal，如果你在之后安装其他包时重新自动安装了triton，你需要重复上述过程
