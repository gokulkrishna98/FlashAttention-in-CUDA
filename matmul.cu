#include <cuda_runtime.h>
#include <algorithm>

__global__
void matmul(float *a, float *b, float *out, int ha, int wa, int hb, int wb, float sf){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    // wa = hb
    if(x < ha && y < wb){
        float res = 0.0f;
        for(int i=0; i<wa; i++){
            res += a[x*wa + i] * b[i*wb + y];
        }
        out[x*wb + y] = (res * sf);
    }
    return;
}



extern "C" void launch_matmul(float *a, float *b, float *out, int ha, int wa, int hb, int wb, float sf = 1.0f) {
    // ha x wb
    int block_size_x = std::min(32, ha);
    int block_size_y = std::min(32, wb);

    dim3 block_size(block_size_x, block_size_y);
    dim3 grid_size((ha + block_size_x -1)/block_size_x, (wb + block_size_y -1)/block_size_y);
    matmul<<<grid_size, block_size>>>(a, b, out, ha, wa, hb, wb, sf);
    return;
}