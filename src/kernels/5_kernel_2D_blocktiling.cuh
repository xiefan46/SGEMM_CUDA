#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// TODO: 测试如果没有__launch_bounds__是否有影响

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(CEIL_DIV(BN, TN) * CEIL_DIV(BM, TM), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    __shared__ float smem_a[BM][BK];
    __shared__ float smem_b[BK][BN];
    float reg_c[TM][TN];
//    for (int i = 0; i < TM; i++) {
//      for (int j = 0; j < TN; j++) {
//        reg_c[i][j] = 0;
//      }
//    }

    for (int bk = 0; bk < K; bk += BK) {
        // load data to shared mem
        const int a_bx = bk;
        const int a_by = blockDim.y * blockIdx.y;
        const int b_bx = blockDim.x * blockIdx.x;
        const int b_by = bk;
        const int a_tx = threadIdx.x;
        const int a_ty = TM * threadIdx.y;
        const int b_tx = TN * threadIdx.x;
        const int b_ty = threadIdx.y;
        #pragma unroll
        for (int i = 0; i < TM; i++) {
          smem_a[a_ty + i][a_tx] = A[(a_by + a_ty + i) * K + a_bx + a_tx];
        }
        #pragma unroll
        for (int i = 0; i < TN; i++) {
          smem_b[b_ty][b_tx + i] = B[(b_by + b_ty) * N + b_bx + b_tx + i];
        }
        __syncthreads();

        // load data to registers
        float reg_a[TM];
        float reg_b[TN];

        #pragma unroll
        for (int i = 0; i < TM; i++) {
          reg_a[i] = smem_a[a_ty + i][a_tx];
        }
        #pragma unroll
        for (int i = 0; i < TN; i++) {
          reg_b[i] = smem_b[b_ty][b_tx + i];
        }
        #pragma unroll
        for (int i = 0; i < TM; i++) {
          for (int j = 0; j < TN; j++) {
            reg_c[i][j] += reg_a[i] * reg_b[j];
          }
        }
        __syncthreads();
    }

    // Write register result to C
    const int c_offset_x = blockDim.x * blockIdx.x + threadIdx.x * TN;
    const int c_offset_y = blockDim.y * blockIdx.y + threadIdx.y * TM;
    #pragma unroll
    for (int i = 0; i < TM; i++) {
      for (int j = 0; j < TN; j++) {
        const int index = (c_offset_y + i) * N + c_offset_x + j;
        C[index] = C[index] * beta + alpha * reg_c[i][j];
      }
    }
}