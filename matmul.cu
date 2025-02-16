#include <cuda_runtime.h>

__global__
void matmul(float *a, float *b, float *out, int ha, int wa, int hb, int wb, float sf){

    // threads along x axis are group into warp
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    // matmul cond: wa = hb
    // for memory coalescing the x should point to column and y to row.
    // out[y, x] = row[y] * col[x];
    if(x < wb && y < ha){
        float res = 0.0f;
        for(int i=0; i<wa; i++){
            res += a[y*wa + i] * b[i*wb + x];
        }
        out[y*wb + x] = (res * sf);
    }
    return;
}



extern "C" void launch_matmul(float *a, float *b, float *out, int ha, int wa, int hb, int wb, float sf = 1.0f) {
    // ha x wb
    int block_size_x = 32;
    int block_size_y = 32;

    dim3 block_size(block_size_x, block_size_y);
    dim3 grid_size((wb + block_size_x -1)/block_size_x, (ha + block_size_y -1)/block_size_y);
    matmul<<<grid_size, block_size>>>(a, b, out, ha, wa, hb, wb, sf);
    return;
}