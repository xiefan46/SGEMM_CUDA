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

  const uint bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;

  const uint THREAD_NUM_PER_BLOCK = blockDim.x * blockDim.y;
  assert(THREAD_NUM_PER_BLOCK == (BM * BN) / (TM * TN));

  const uint TOTAL_ELEMENT_BLOCK_A = BM * BK;
  const uint TOTAL_ELEMENT_BLOCK_B = BN * BK;
  const uint ELEMENT_PER_THREAD_A = TOTAL_ELEMENT_BLOCK_A / THREAD_NUM_PER_BLOCK;
  const uint ELEMENT_PER_THREAD_B = TOTAL_ELEMENT_BLOCK_B / THREAD_NUM_PER_BLOCK;

  __shared__ float smem_a[BK][BM];
  __shared__ float smem_b[BK][BN];
  float reg_c[TM][TN] = {0};
  float reg_a[TM] = {0};
  float reg_b[TN] = {0};


  A += by * BM * K;
  B += bx * BN;
  C += by * BM * N + bx * BN;

  assert(ELEMENT_PER_THREAD_A % 4 == 0);
  assert(ELEMENT_PER_THREAD_B % 4 == 0);

  const uint inner_off_a = (ty * blockDim.x + tx) * ELEMENT_PER_THREAD_A;
  const uint inner_off_b = (ty * blockDim.x + tx) * ELEMENT_PER_THREAD_B;

  for (uint k = 0; k < K; k += BK) {
    // A += k;
    // B += N * k;
	uint inner_row_a = inner_off_a / BK;
    uint inner_col_a = inner_off_a % BK;
    uint smem_row_a = inner_col_a;
    uint smem_col_a = inner_row_a;
    for (uint i = 0; i < ELEMENT_PER_THREAD_A; i++) {
      // smem_a[smem_row_a + i][smem_col_a] = inner_off_a + i < BM * BK ? A[k + inner_row_a * K + inner_col_a + i] : 0;
      smem_a[smem_row_a + i][smem_col_a] = A[k + inner_row_a * K + inner_col_a + i];
    }

    uint inner_row_b = inner_off_b / BN;
    uint inner_col_b = inner_off_b % BN;
    for (uint i = 0; i < ELEMENT_PER_THREAD_B; i++) {
      // smem_b[inner_row_b][inner_col_b + i] = inner_off_b + i < BN * BK ? B[N * k + inner_row_b * N + inner_col_b + i] : 0;
       smem_b[inner_row_b][inner_col_b + i] = B[N * k + inner_row_b * N + inner_col_b + i];
    }

    __syncthreads();

    for (uint tk = 0; tk < BK; tk++) {
      for (int i = 0; i < TM; i++) {
        //reg_a[i] = i + TM * ty < BM ? smem_a[tk][TM * ty + i] : 0;
        reg_a[i] = smem_a[tk][TM * ty + i];
      }

      for (uint i = 0; i < TN; i++) {
        //reg_b[i] = i + TN * tx < BN ? smem_b[tk][TN * tx + i] : 0;
        reg_b[i] = smem_b[tk][TN * tx + i];
      }

      for (uint i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
          reg_c[i][j] += reg_a[i] * reg_b[j];
        }
      }
    }
    __syncthreads();
  }

  for (uint i = 0; i < TM; i++) {
    for (uint j = 0; j < TN; j++) {
      uint row_c = ty * TM + i;
      uint col_c = tx * TN + j;
//      if (row_c < BM && col_c < BN) {
//        C[row_c * N + col_c] = beta * C[row_c * N + col_c] + alpha * reg_c[i][j];
//      }
       C[row_c * N + col_c] = beta * C[row_c * N + col_c] + alpha * reg_c[i][j];
    }
  }
}


//template <const int BM, const int BN, const int BK, const int TM, const int TN>
//__global__ void sgemmVectorize(const int M, const int N, const int K, float alpha, float *A,
//                               float *B, float beta, float *C) {
//
//  const int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
//
//  const int THREAD_NUM_PER_BLOCK = blockDim.x * blockDim.y;
//  assert(THREAD_NUM_PER_BLOCK == (BM * BN) / (TM * TN));
//
//  const int TOTAL_ELEMENT_BLOCK_A = BM * BK;
//  const int TOTAL_ELEMENT_BLOCK_B = BN * BK;
//  const int ELEMENT_PER_THREAD_A = TOTAL_ELEMENT_BLOCK_A / THREAD_NUM_PER_BLOCK;
//  const int ELEMENT_PER_THREAD_B = TOTAL_ELEMENT_BLOCK_B / THREAD_NUM_PER_BLOCK;
//
//  __shared__ float smem_a[BK][BM];
//  __shared__ float smem_b[BK][BN];
//  float reg_c[TM][TN] = {0};
//
//
//  A += by * BM * K;
//  B += bx * BN;
//  C += by * BM * N + bx * BN;
//
//  assert(ELEMENT_PER_THREAD_A % 4 == 0);
//  assert(ELEMENT_PER_THREAD_B % 4 == 0);
//
//  int inner_off_a = (ty * blockDim.x + tx) * ELEMENT_PER_THREAD_A;
//  int inner_off_b = (ty * blockDim.x + tx) * ELEMENT_PER_THREAD_B;
//
//  for (int k = 0; k < K; k += BK) {
//    // A += k;
//    // B += N * k;
//	int inner_row_a = inner_off_a / BK;
//    int inner_col_a = inner_off_a % BK;
//    int smem_row_a = inner_col_a;
//    int smem_col_a = inner_row_a;
//    for (int i = 0; i < ELEMENT_PER_THREAD_A; i++) {
//      smem_a[smem_row_a + i][smem_col_a] = inner_off_a + i < BM * BK ? A[k + inner_row_a * K + inner_col_a + i] : 0;
//    }
//
//    int inner_row_b = inner_off_b / BN;
//    int inner_col_b = inner_off_b % BN;
//    for (int i = 0; i < ELEMENT_PER_THREAD_B; i++) {
//      smem_b[inner_row_b][inner_col_b + i] = inner_off_b + i < BN * BK ? B[N * k + inner_row_b * N + inner_col_b + i] : 0;
//    }
//
//    __syncthreads();
//    float reg_a[TM] = {0};
//    float reg_b[TN] = {0};
//    for (int tk = 0; tk < BK; tk++) {
//      for (int i = 0; i < TM; i++) {
//        reg_a[i] = i + TM * ty < BM ? smem_a[tk][TM * ty + i] : 0;
//      }
//
//      for (int i = 0; i < TN; i++) {
//        reg_b[i] = i + TN * tx < BN ? smem_b[tk][TN * tx + i] : 0;
//      }
//
//      for (int i = 0; i < TM; i++) {
//        for (int j = 0; j < TN; j++) {
//          reg_c[i][j] += reg_a[i] * reg_b[j];
//        }
//      }
//    }
//    __syncthreads();
//  }
//
//  for (int i = 0; i < TM; i++) {
//    for (int j = 0; j < TN; j++) {
//      int row_c = ty * TM + i;
//      int col_c = tx * TN + j;
//      if (row_c < BM && col_c < BN) {
//        C[row_c * N + col_c] = beta * C[row_c * N + col_c] + alpha * reg_c[i][j];
//      }
//    }
//  }
//}


