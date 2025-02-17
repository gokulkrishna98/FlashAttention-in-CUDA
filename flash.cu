#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>

/*
Assumption:
    - We are operating on 2d space (ignoring batch and num_heads)
    - Width of Tiles = d
    - Height of Tiles = Bc = Br = B
    - Number of blocks = 1
    - Number of Tiles per matrix = T
    - Number of threads per block = B = Each threads loads each row (withing the
tile)
    - TODO:
    -   move l, m, o to sram.
*/
__global__ void flash_attention(float *Q, float *K, float *V, int N, int d,
                                int T, int B, float softmax_scale, float *L,
                                float *M, float *O) {
  int tid = threadIdx.x;

  // shared memory
  extern __shared__ float sram[];
  int tile_size = B * d;
  float *Qi = sram;
  float *Kj = &sram[tile_size];
  float *Vj = &sram[tile_size * 2];
  float *S = &sram[tile_size * 3];

  for (int j = 0; j < T; j++) {
    // Loading Kj, Vj into SRAM
    for (int k = 0; k < d; k++) {
      Kj[tid * d + k] = K[(j*tile_size) + (tid*d) + k];
      Vj[tid * d + k] = V[(j*tile_size) + (tid*d) + k];
    }
    __syncthreads();
    // Loading Qi into SRAM
    for (int i = 0; i < T; i++) {
      for (int k = 0; k < d; k++) {
        Qi[(tid*d) + k] = Q[(i*tile_size) + (tid*d) + k];
      }
      // No need for sync thread, because each row is tied to tid as is accessed
      // independently
      float li = L[(B*i) + tid];
      float mi = M[(B*i) + tid];

      // The max element across row given by tid
      float m = -FLT_MAX;
      for (int y = 0; y < B; y++) {
        float sum = 0.0f;
        for (int x = 0; x < d; x++) {
          sum += Qi[(tid*d) + x] * Kj[(y*d) + x];
        }
        sum = sum * softmax_scale;
        S[(tid*B) + y] = sum;
        if (sum > m) {
          m = sum;
        }
      }

      // computing l_new
      float l = 0.0f;
      for (int y = 0; y < B; y++) {
        S[(B * tid) + y] = expf(S[(B*tid) + y] - m);
        l += S[(B * tid) + y];
      }

      // Compute new m and l
      float m_new = max(mi, m);
      float l_new = (expf(mi - m_new) * li) + (expf(m - m_new) * l);

      // writing result back
      for (int x = 0; x < d; x++) {
        float pv = 0; // Pij * Vj
        for (int y = 0; y < B; y++) {
          pv += S[(B * tid) + y] * Vj[(y * d) + x];
        }
        O[(tile_size * i) + (tid * d) + x] =
            (1 / l_new) *
            ((li * expf(li - m_new) * O[(tile_size * i) + (tid * d) + x]) +
             (expf(m - m_new) * pv));
      }
      M[(B * i) + tid] = m_new;
      L[(B * i) + tid] = l_new;
    }
    __syncthreads();
  }
}

// As of now we support B = 32. Make sure Height of the matrix (N is multiple of
// 32)
extern "C" void launch_flash(float *Q, float *K, float *V, float *O, float *M,
                             float *L, int N, int d, int T, int B = 32,
                             float sf = 1.0f) {
  // ha x wb
  dim3 block_size(32);
  dim3 grid_size(1);
  int sram_size = (3 * B * d + B * B) * sizeof(float);
  flash_attention<<<grid_size, block_size, sram_size>>>(Q, K, V, N, d, T, B, sf,
                                                        L, M, O);
  return;
}