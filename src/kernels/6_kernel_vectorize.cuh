#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmVectorize(const int M, const int N, const int K, float alpha, float *A,
                               float *B, float beta, float *C) {
  const int THREAD_NUM_PER_BLOCK = blockDim.x * blockDim.y;
  assert(THREAD_NUM_PER_BLOCK == (BM * BN) / (TM * TN));

  const int TOTAL_ELEMENT_BLOCK_A = BM * BK;
  const int TOTAL_ELEMENT_BLOCK_B = BN * BK;
  const int ELEMENT_PER_THREAD_A = TOTAL_ELEMENT_BLOCK_A / THREAD_NUM_PER_BLOCK;
  const int ELEMENT_PER_THREAD_B = TOTAL_ELEMENT_BLOCK_B / THREAD_NUM_PER_BLOCK;

  __shared__ float smem_a[BK][BM];
  __shared__ float smem_b[BK][BN];
  float reg_c[TM][TN] = {0};

  for (int k = 0; k < K; k += BK) {

    // load data to smem
    int global_offset_a = blockIdx.y * BM * K + k;
	int block_offset_a = (threadIdx.y * blockDim.x + threadIdx.x) * ELEMENT_PER_THREAD_A;
    int row_off_a = block_offset_a / BM;
    int col_off_a = block_offset_a % BM;
    #pragma unroll
    for (int i = 0; i < ELEMENT_PER_THREAD_A; i++) {
      assert(row_off_a + i < BK);
      assert(col_off_a < BM);
      if (global_offset_a + block_offset_a + i < M * K) {
        smem_a[row_off_a + i][col_off_a] = A[global_offset_a + block_offset_a + i];
      } else {
        smem_a[row_off_a + i][col_off_a] = 0.0;
      }
    }

    int global_offset_b = k * N + blockIdx.y * BN;
	int block_offset_b = (threadIdx.y * blockDim.x + threadIdx.x) * ELEMENT_PER_THREAD_B;
    int row_off_b = block_offset_b / BN;
    int col_off_b = block_offset_b % BN;
    #pragma unroll
    for (int i = 0; i < ELEMENT_PER_THREAD_B; i++) {
      assert(row_off_b < BK);
      assert(col_off_b + i < BN);
      if (global_offset_b + block_offset_b + i < K * N) {
        smem_b[row_off_b][col_off_b + i] = B[global_offset_b + block_offset_b + i];
      } else {
        smem_b[row_off_b][col_off_b + i] = 0.0;
      }
    }
    __syncthreads();

    //load smem to reg
    float reg_a[TM] = {0};
    float reg_b[TN] = {0};

    for (int tk = 0; tk < BK; tk++) {
      for (int i = 0; i < TM; i++) {
        int idx = TM * threadIdx.y + i;
        reg_a[i] = idx < BM ? smem_a[tk][idx] : 0;
      }
      for (int i = 0; i < TN; i++) {
        int idx = TN * threadIdx.x + i;
        reg_b[i] = idx < BN ? smem_b[tk][idx] : 0;
      }
      for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
          reg_c[i][j] += reg_a[i] * reg_b[j];
        }
      }
    }
    __syncthreads();
  }
  for (int i = 0; i < TM; i++) {
      for (int j = 0; j < TN; j++) {
        int row_c = blockIdx.y * BM + threadIdx.y * TM + i;
        int col_c = blockIdx.x * BN + threadIdx.x * TN + j;
       	int idx = row_c * N + col_c;
        assert(idx < M * N);
        C[idx] = C[idx] * beta + reg_c[i][j] * alpha;
      }
    }
}