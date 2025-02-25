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
    const int global_x = bx * BLOCKSIZE + tx;
    const int global_y = by * BLOCKSIZE + ty;
	smem_a[ty][tx] = 0;
    smem_b[ty][tx] = 0;


    float tmp_value = 0;

   if (global_x >= N || global_y >= M) {
     return;
   }
  	for (int b = 0; b < K; b += BLOCKSIZE) {
    	smem_a[ty][tx] = tx + b < K ? A[global_y * K + tx + b] : 0;
    	smem_b[ty][tx] = ty + b < K ? B[(ty + b) * N + global_x] : 0;
    	__syncthreads();
    	// 线程 tx, ty需要拿出 smem_a 的第ty行以及smem_b的tx列
        int valid_k = min(BLOCKSIZE, K - b);
    	for (int k = 0; k < valid_k; k++) {
        	tmp_value += smem_a[ty][k] * smem_b[k][tx];
    	}
    	__syncthreads();
   	}
    C[global_y * N + global_x] = C[global_y * N + global_x] * beta + alpha * tmp_value;

}