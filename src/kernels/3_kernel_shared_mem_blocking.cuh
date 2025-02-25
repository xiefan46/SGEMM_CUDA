#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))


// 多个线程可以共享share memory 中的数据
// 每个线程处理C中的一个元素
// TODO: 探索bank conflict影响
template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
    __shared__ float smem_a[BLOCKSIZE][BLOCKSIZE];
    __shared__ float smem_b[BLOCKSIZE][BLOCKSIZE];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int global_x = bx * blockDim.x + tx;
    const int global_y = by * blockDim.y + ty;

    if (global_y < M && global_x < N) {
    	float tmp_value = 0;
    	for (int b = 0; b < K; b += BLOCKSIZE) {
      	smem_a[ty][tx] = A[global_y * K + global_x + b];
      	smem_b[ty][tx] = B[(global_y + b) * N + global_x];
      	__syncthreads();
      	// 线程 tx, ty需要拿出 smem_a 的第ty行以及smem_b的tx列
      	for (int k = 0; k < BLOCKSIZE; k++) {
        	tmp_value += smem_a[ty][k] * smem_b[k][tx];
      	}
      	__syncthreads();
    	}
    	C[global_y * N + global_x] = C[global_y * N + global_x] * beta + alpha * tmp_value;
   	}
}