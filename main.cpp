#include <cmath>
#include <cstdint>
#include <torch/torch.h>
#include <cuda_runtime.h>
// #include <iostream>
#include <utility>


extern "C" void launch_transpose(float *in, float *out, int E, int S);

/*
 * We are assuming batch size = 1
 * We have 3 tensors:
 *   - Query: (L, E)
 *   - Key: (S, E)
 *   - Value: (S, E_v)
 *
 * Attention Computation:
 *  1. Compute similarity scores (s) = Q @ transpose(K)
 *  2. Compute attention score (a) = softargmax(s)
 *  3. Compute Attention weight = a @ V
 *
 * Note: shape of s/a = (target_len, source_len)
 */

 /*
    TODO Learn:
    - 1. Review Grid, Block
    - 2. Review the slides regarding performance improvement concepts
    -   a. Tiling and Memory Coelescing.
    -   b. Reduction.
    -   c. Shared Memory. 
    - 3. Performance checklist.
 */

/*
    Transpose Kernel
*/ 


torch ::Tensor pytorch_attention(const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &v){
    int64_t head_dim = q.size(-1);
    double scale_factor = 1.0 / std::sqrt(head_dim);

    auto similarity = torch::matmul(q, k.transpose(-2, -1)) * scale_factor;
    auto attn_weights = torch::softmax(similarity, -1);
    auto output = torch::matmul(attn_weights, v);
    return output;
}

void standard_attention(const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &v){
    float *q_ptr = q.contiguous().data_ptr<float>();
    float *k_ptr = k.contiguous().data_ptr<float>();
    float *v_ptr = v.contiguous().data_ptr<float>();

    float *dev_q, *dev_k, *dev_v;
    float *dev_i1, *dev_i2, *dev_i3, *dev_out;


    auto q_shape = std::make_pair(q.size(0), q.size(1));
    auto k_shape = std::make_pair(k.size(0), k.size(1));
    auto v_shape = std::make_pair(v.size(0), v.size(1));

    int L = q.size(0); int E = q.size(1);
    int S = k.size(0); int E_v = k.size(1);

    float *temp = (float *)malloc(sizeof(float)*S*E);
    /*
        All the data is allocated before hand. The computation is as follows
        Q = (L, E), K = (S, E), V = (S, E_v)
        I1 = transpose(K) --- (E, S)
        I2 = Q @ I1 --- (L, S)
        I3 = softmax(I2) --- (L, S) 
        out = I3 @ V --- (L, E_v)
    */

    // Allocating query, key and values
    cudaMalloc((void**)&dev_q, q.numel()*sizeof(float));
    cudaMalloc((void**)&dev_k, k.numel()*sizeof(float));
    cudaMalloc((void**)&dev_v, v.numel()*sizeof(float));

    // Allocating intermediate results and output 
    cudaMalloc((void**)&dev_i1, (E*S)*sizeof(float));
    cudaMalloc((void**)&dev_i2, (L*S)*sizeof(float));
    cudaMalloc((void**)&dev_i3, (L*S)*sizeof(float));
    cudaMalloc((void**)&dev_out, (L*E_v)*sizeof(float));

    // Copying form CPU to GPU (q, k & v)
    cudaMemcpy(dev_q, q_ptr, q.numel()*sizeof(float),cudaMemcpyHostToDevice); 
    cudaMemcpy(dev_k, k_ptr, k.numel()*sizeof(float),cudaMemcpyHostToDevice); 
    cudaMemcpy(dev_v, v_ptr, v.numel()*sizeof(float),cudaMemcpyHostToDevice); 

    /* Tranpose of K */
    launch_transpose(dev_k, dev_i1, E, S);

    cudaMemcpy(temp, dev_i1, sizeof(float)*S*E, cudaMemcpyDeviceToHost);

    // for(int i=0; i<E; i++){
    //     for(int j=0; j<S; j++){
    //         int index = i*S + j;
    //         std::cout << temp[index] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Freeing allocated device memory
    cudaFree(dev_q);
    cudaFree(dev_k);
    cudaFree(dev_v);
    cudaFree(dev_i1);
    cudaFree(dev_i2);
    cudaFree(dev_i3);
    cudaFree(dev_out);
    return;
}

void flash_attention(){
    return;
}

int main(){
    torch::manual_seed(42);

    // // Define the input tensors (query, key, value)
    int64_t L = 3;
    int64_t S = 10000;
    int64_t E = 10000;
    int64_t E_v = 3;

    // // Create random tensors for query, key, and value
    const torch::Tensor query = torch::randn({L, E});
    const torch::Tensor key = torch::randn({S,E});
    const torch::Tensor value = torch::randn({S, E_v});
    

    float *dev_q, *dev_k, *dev_v;
    // std::cout << "Key Value\n" << key << std::endl;

    // // Compute scaled dot-product attention
    torch::Tensor attn_output = pytorch_attention(query, key, value);
    // standard_attention(query, key, value);

    // // Print the output
    // std::cout << "Attention Output: " << attn_output << std::endl;

    return 0;
}