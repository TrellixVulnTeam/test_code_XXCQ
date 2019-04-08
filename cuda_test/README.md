# CUDA Installation and Usage

## Install CUDA

Follow the official link:





## Set up CUDA environment


### Set up gcc/g++ version

Ref: https://stackoverflow.com/questions/17275348/how-to-specify-new-gcc-path-for-cmake

Different CUDA version supports corresponding gcc or g++ version. For instance, CUDA 9 only supports gcc version NO LATER THAN 7. Therefore, if your default gcc compiler is too new, you may get an error about the gcc version like this when building your CUDA code:
```
 #error -- unsupported GNU version! gcc versions later than 7 are not supported!
```

To solve this, you may switch your default gcc and g++ in one of your terminal environment (so that it doesn't inflence the others) like this:
```shell
export CC=/usr/bin/gcc-7
export CXX=/usr/bin/g++-7
```
Then run cmake and build your code. Check the C compiler version in your cmake prompt to see if it works.

### Keywords

#### \_\_global\_\_ vs. \_\_device\_\_
Global functions are also called "kernels". It's the functions that you may call from the host side using CUDA kernel call semantics (<<<...>>>).

Device functions can only be called from other device or global functions. **\_\_device\_\_** functions cannot be called from host code.

