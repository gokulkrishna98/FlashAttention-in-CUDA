#include <algorithm>
#include <cfloat>
#include <cstdio>

/*
    General algorithm
    -   1. Finding max using reduction and shared memory
    -   2. Finding exp and sum using reduction and shared memory
    -   3. compute the softmax per thread. 
*/
__global__
void softmax(float *in, float *out, int h, int w){
    __shared__ float tile[1024];
    int x = blockIdx.x;
    int y = threadIdx.x;
    
    /* computing max using reduction and shared memory */
    float max_v = -FLT_MAX;
    for(int i = y; i < w; i += blockDim.x){
        if(i < w){
            max_v = fmaxf(max_v, in[x*w + i]);
        }
    }

    tile[y] = max_v;
    __syncthreads();
    for(int stride = blockDim.x/2; stride > 0; stride = stride/2){
        if(y < stride){
            tile[y] = fmaxf(tile[y], tile[y + stride]);
        }
        __syncthreads();
    }
    max_v = tile[0];
    
    /* computing exp and sum using reduction and shared memory */
    float sum = 0.0f;
    for (int i = y; i < w; i += blockDim.x) {
        if(i < w){
            out[x*w + i] = expf(in[x*w + i] - max_v);
            sum += out[x*w + i];
        }
    }
    tile[y] = sum; 
    __syncthreads();
    for(int stride = blockDim.x/2; stride > 0; stride /= 2){
        if(y < stride){
            tile[y] += tile[y + stride];
        }
        __syncthreads();
    }
    sum = tile[0];
    for(int i=y; i < w; i+= blockDim.x){
        if(i < w){
          out[x * w + i] /= sum;
        }
    }
    return;
}

extern "C" void launch_softmax(float *in, float *out, int h, int w) {
    int block_size_x = 1024;
    dim3 block_size(block_size_x);
    dim3 grid_size(h);
    softmax<<<grid_size, block_size>>>(in, out, h, w);
}