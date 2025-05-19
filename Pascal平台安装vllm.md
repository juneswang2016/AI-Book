## Pascal平台(SM60)安装vllm 0.85

L
# 编译

首先你需要有一份vllm 0.8.5的项目代码
'''
git clone -b 分支名 --single-branch 仓库UR
'''

打开项目根目录下的CMakeLists.txt，找到34行的CUDA_SUPPORTED_ARCHS，并加上6.0;6.1

最后看起来应该像下面这样：

# Supported NVIDIA architectures.

set(CUDA_SUPPORTED_ARCHS "6.0;6.1;7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0")

然后就可以愉快的开始编译了，然后喜提一大串报错：

no instance of overloaded function "atomicAdd" matches the argument list

NV原话：

The 16-bit __half  floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.
翻译：vllm的cuda代码用了half精度的atomicAdd，然而pascal架构的计算能力并不支持这个操作

现在可以把你的pascal设备丢进垃圾桶并换成高级的B200了

找到项目目录下csrc/moe/moe_wna16.cu，添加以下代码：

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

问题解决。不出意外pip install -e .可以正确编译安装。

运行

https://github.com/sasha0552/pascal-pkgs-ci
现在看着一切都很美好，但是如果你现在直接运行你的pascal gpu就会报错然后被pytorch告知pascal死了

解决办法：终端运行

sed -e "s/.major < 7/.major < 6/g"                                 \

    -e "s/.major >= 7/.major >= 6/g"                               \

    -i                                                             \

    venv/lib/python3.10/site-packages/torch/_inductor/scheduler.py \

    venv/lib/python3.10/site-packages/torch/utils/_triton.py

记得把路径改成你的python环境和版本

继续运行，然后你的pascal设备会被triton判死刑

解决办法：删除triton，下载专门调教成适配pascal架构的triton3.2.0版本，链接：

https://sasha0552.github.io/pascal-pkgs-ci/

不建议下载此站预编译好的vllm-pascal，可能会出现神秘问题

注：一定要在删除triton后再安装triton-pascal，如果你在之后安装其他包时重新自动安装了triton，你需要重复上述过程

按正常，到这里一切都应该可以正常运行了，但是up又遇到了一个神秘问题(如果你没有遇到就不需要接着看下去)：

ERROR 05-09 17:35:49 [engine.py:160] triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 65536, Hardware limit: 49152. Reducing block sizes or `num_stages` may help.

非常明显，我们需要降低block size或者num stages，但是我要上哪找这两个玩意啊



找到python文件夹下site-packages/vllm/attention/ops/prefix_prefill.py 903行，把BLOCK_M BLOCK_N各降低一半

虽然我不知道这两个参数是用来干什么的，但是就结果来看，运行地很好

结束
