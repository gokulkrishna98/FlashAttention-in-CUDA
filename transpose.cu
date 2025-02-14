#include <algorithm>

__global__
void transpose_kernel(float *in, float *out, int height, int width){
    __shared__ float tile[32][32];
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x < height && y < width){
        tile[threadIdx.x][threadIdx.y] = in[y*height + x];
    }
    __syncthreads();

    if(x < height && y < width){
        out[x*width + y] = tile[threadIdx.x][threadIdx.y];
    }
    return;
}

// height and width of out matrix
extern "C" void launch_transpose(float *in, float *out, int height, int width) {
    int block_size_x = std::min(32, height); 
    int block_size_y = std::min(32, width);
    dim3 block_size(block_size_x, block_size_y);
    dim3 grid_size((height + block_size_x - 1)/block_size_x, (width + block_size_y - 1)/block_size_y);
    transpose_kernel<<<grid_size, block_size>>>(in, out, height, width);
}
