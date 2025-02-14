#include <torch/torch.h>
#include <cuda_runtime.h>
/*
 * We are assuming batch size = 1
 * We have 3 tensors:
 *   - Query: (target_len, dk)
 *   - Key: (source_len, dk)
 *   - Value: (source_len, dv)
 *
 * Attention Computation:
 *  1. Compute similarity scores (s) = Q @ transpose(K)
 *  2. Compute attention score (a) = softargmax(s)
 *  3. Compute Attention weight = a @ V
 *
 * Note: shape of s/a = (target_len, source_len)
 */
int main(){

    return 0;
}