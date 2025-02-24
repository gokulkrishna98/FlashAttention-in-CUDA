#include <cmath>
#include <cstdint>
#include <torch/torch.h>
#include <cuda_runtime.h>
// #include <iostream>
#include <utility>


extern "C" void launch_transpose(float *in, float *out, int E, int S);
extern "C" void launch_matmul(float *a, float *b, float *out, int ha, int wa, int hb, int wb, float sf = 1.0f);
extern "C" void launch_softmax(float *in, float *out, int h, int w);
extern "C" void launch_flash(float *Q, float *K, float* V, float *O, float *M, float*L, int N, int d, int T, int B = 32, float sf = 1.0f); 

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
 bool is_equal(const float* data, torch::Tensor tensor, int size, float rtol = 1e-3, float atol = 1e-3) {
    torch::Tensor data_tensor = torch::from_blob((void*)data, {size}, torch::kFloat32);
    tensor = tensor.view({size}); 
    return torch::allclose(data_tensor, tensor, rtol, atol);
}


torch ::Tensor pytorch_attention(const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &v){
    int64_t head_dim = q.size(-1);
    double scale_factor = 1.0 / std::sqrt(head_dim);
    auto similarity = torch::matmul(q, k.transpose(-2, -1)) * scale_factor;
    auto attn_weights = torch::softmax(similarity, -1);
    auto output = torch::matmul(attn_weights, v);
    return output;
}

void flash_attention(const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &v){
    float *q_ptr = q.contiguous().data_ptr<float>();
    float *k_ptr = k.contiguous().data_ptr<float>();
    float *v_ptr = v.contiguous().data_ptr<float>();

    float *dev_q, *dev_k, *dev_v;
    float *dev_l, *dev_m, *dev_i3, *dev_out;


    auto q_shape = std::make_pair(q.size(0), q.size(1));
    auto k_shape = std::make_pair(k.size(0), k.size(1));
    auto v_shape = std::make_pair(v.size(0), v.size(1));

    int L = q.size(0); int E = q.size(1);
    int S = k.size(0); int E_v = k.size(1);

    float *out = (float *)malloc(sizeof(float)*L*E_v);

    cudaMalloc((void**)&dev_q, q.numel()*sizeof(float));
    cudaMalloc((void**)&dev_k, k.numel()*sizeof(float));
    cudaMalloc((void**)&dev_v, v.numel()*sizeof(float));
    cudaMalloc((void**)&dev_out, (L*E_v)*sizeof(float));

    cudaMemcpy(dev_q, q_ptr, q.numel()*sizeof(float),cudaMemcpyHostToDevice); 
    cudaMemcpy(dev_k, k_ptr, k.numel()*sizeof(float),cudaMemcpyHostToDevice); 
    cudaMemcpy(dev_v, v_ptr, v.numel()*sizeof(float),cudaMemcpyHostToDevice); 

    cudaMalloc((void**)&dev_l, (L)*sizeof(float));
    cudaMalloc((void**)&dev_m, (L)*sizeof(float));

    launch_flash(dev_q, dev_k, dev_v, dev_out, dev_m, dev_l, L, E, L/32, 32, 1/std::sqrt(E)); 

    cudaMemcpy(out, dev_out, sizeof(float)*L*E_v, cudaMemcpyDeviceToHost);

    auto torch_out = pytorch_attention(q, k, v);
    std::cout << "Is same? :- " << is_equal(out, torch_out, L*E_v) << std::endl;
    // for(int i=0; i<L; i++){
    //     for(int j=0; j<E_v; j++){
    //         std::cout << out[i*E_v + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "Output: \n" << torch_out << std::endl;

    free(out);
    cudaFree(dev_q);
    cudaFree(dev_k);
    cudaFree(dev_v);
    cudaFree(dev_l);
    cudaFree(dev_m);
    cudaFree(dev_out);

    return;
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

    // @cleanup: for printing stuff, remove it later
    float *out = (float *)malloc(sizeof(float)*L*E_v);
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

    /* Q @ K^T */
    float scale_factor = 1.0 / std::sqrt(E);
    launch_matmul(dev_q, dev_i1, dev_i2, L, E, E, S, scale_factor);

    /* softmax */
    launch_softmax(dev_i2, dev_i3, L, S);

    /* I3 @ V*/
    launch_matmul(dev_i3, dev_v, dev_out, L, S, S, E_v, 1.0f);
    cudaMemcpy(out, dev_out, sizeof(float)*L*E_v, cudaMemcpyDeviceToHost);

    /* Comparing the my implementation with pytorch cpu values */
    auto torch_out = pytorch_attention(q, k, v);
    std::cout << "Is same? :- " << is_equal(out, torch_out, L*E_v) << std::endl;

    /*
    printing value
    */

    // Freeing allocated device memory
    free(out);
    cudaFree(dev_q);
    cudaFree(dev_k);
    cudaFree(dev_v);
    cudaFree(dev_i1);
    cudaFree(dev_i2);
    cudaFree(dev_i3);
    cudaFree(dev_out);
    return;
}

// @todo 
void flash_attention(){
    return;
}

int main(){
    torch::manual_seed(42);

    // Shapes (var names based on pytorch: scaled_dot_produce_attention)
    int64_t L = 32; 
    int64_t S = 32;
    int64_t E = 4; 
    int64_t E_v = 4;

    // more robust values
    // int64_t L = 117; 
    // int64_t S = 1287;
    // int64_t E = 1333; 
    // int64_t E_v = 1333;

    // Create random tensors for query, key, and value
    const torch::Tensor query = torch::randn({L, E});
    const torch::Tensor key = torch::randn({S,E});
    const torch::Tensor value = torch::randn({S, E_v});
    

    float *dev_q, *dev_k, *dev_v;

    // Compute scaled dot-product attention
    // standard_attention(query, key, value);
    flash_attention(query, key, value);
    return 0;
}