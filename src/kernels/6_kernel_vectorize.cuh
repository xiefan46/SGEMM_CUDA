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

  const int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;

  const int THREAD_NUM_PER_BLOCK = blockDim.x * blockDim.y;
  assert(THREAD_NUM_PER_BLOCK == (BM * BN) / (TM * TN));

  const int TOTAL_ELEMENT_BLOCK_A = BM * BK;
  const int TOTAL_ELEMENT_BLOCK_B = BN * BK;
  const int ELEMENT_PER_THREAD_A = TOTAL_ELEMENT_BLOCK_A / THREAD_NUM_PER_BLOCK;
  const int ELEMENT_PER_THREAD_B = TOTAL_ELEMENT_BLOCK_B / THREAD_NUM_PER_BLOCK;

  __shared__ float smem_a[BK][BM];
  __shared__ float smem_b[BK][BN];
  float reg_c[TM][TN] = {0};


  A += by * BM * K;
  B += bx * BN;
  C += by * BM * K + bx * BN;

  assert(ELEMENT_PER_THREAD_A % 4 == 0);
  assert(ELEMENT_PER_THREAD_B % 4 == 0);

  int inner_off_a = (ty * blockDim.x + tx) * ELEMENT_PER_THREAD_A;
  int inner_off_b = (ty * blockDim.x + tx) * ELEMENT_PER_THREAD_B;

  float reg_c[TM][TN] = {0};

  for (int k = 0; k < K; k += BK) {
    A += k;
    B += N * k;
	int inner_row_a = inner_off_a / BK;
    int inner_col_a = inner_off_a % BK;
    int smem_row_a = inner_off_a / BM;
    int smem_row_b = inner_off_a % BM;
    for (int i = 0; i < ELEMENT_PER_THREAD_A; i++) {
      smem_a[smem_row_a + i][smem_row_b] = inner_row_a * K + inner_col_a + i < BM * BK ? A[inner_row_a * K + inner_col_a + i] : 0;
    }

    int inner_row_b = inner_off_b / BN;
    int inner_col_b = inner_off_b % BN;
    int smem_row_b = inner_row_b;
    int smem_col_b = inner_col_b;
    for (int i = 0; i < ELEMENT_PER_THREAD_B; i++) {
      smem_b[smem_row_b][smem_col_b] = inner_row_b * N + inner_col_b + i < BN * BK ? B[inner_row_b * N + inner_col_b + i] : 0;
    }

    __syncthreads();
    float reg_a[TM] = {0};
    float reg_b[TN] = {0};
    for (int tk = 0; tk < BK; tk++) {
      for (int i = 0; i < TM; i++) {
        reg_a[i] = i + TM * ty < BM ? smem_a[tk][i + TM * ty] : 0;
      }

      for (int i = 0; i < TN; i++) {
        reg_b[i] = i + TN * tx < BN ? smem_b[tk][i + TN * tx] : 0;
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
      int row_c = ty * TM + i;
      int col_c = tx * TN + j;
      if (row_c < BM && col_c < BN) {
        C[row_c * N + col_c] = beta * C[row_c * N + col_c] + alpha * reg_c[i][j];
      }
    }
  }
}

//template <const int BM, const int BN, const int BK, const int TM, const int TN>
//__global__ void sgemmVectorize(const int M, const int N, const int K, float alpha, float *A,
//                               float *B, float beta, float *C) {
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
//  for (int k = 0; k < K; k += BK) {
//
//    // load data to smem
//    int global_offset_a = blockIdx.y * BM * K + k;
//    #pragma unroll
//    for (int i = 0; i < ELEMENT_PER_THREAD_A; i++) {
//      int block_offset_a = (threadIdx.y * blockDim.x + threadIdx.x) * ELEMENT_PER_THREAD_A + i;
//      int row_off_a = block_offset_a / BM;
//      int col_off_a = block_offset_a % BM;
//      assert(row_off_a < BK);
//      if (global_offset_a + block_offset_a < M * K) {
//        smem_a[row_off_a][col_off_a] = A[global_offset_a + block_offset_a];
//      } else {
//        smem_a[row_off_a][col_off_a] = 0.0;
//      }
//    }
//
//	int global_offset_b = k * N + blockIdx.y * BN;
//    #pragma unroll
//    for (int i = 0; i < ELEMENT_PER_THREAD_B; i++) {
//	  int block_offset_b = (threadIdx.y * blockDim.x + threadIdx.x) * ELEMENT_PER_THREAD_B + i;
//      int row_off_b = block_offset_b / BN;
//      int col_off_b = block_offset_b % BN;
//      assert(row_off_b < BK);
//      if (global_offset_b + block_offset_b < K * N) {
//        smem_b[row_off_b][col_off_b] = B[global_offset_b + block_offset_b];
//      } else {
//        smem_b[row_off_b][col_off_b] = 0.0;
//      }
//    }
//    __syncthreads();
//
//    //load smem to reg
//    float reg_a[TM] = {0};
//    float reg_b[TN] = {0};
//
//    assert(TM * blockDim.y == BM);
//    assert(TN * blockDim.x == BN);
//    for (int tk = 0; tk < BK; tk++) {
//      for (int i = 0; i < TM; i++) {
//        int idx = TM * threadIdx.y + i;
//        reg_a[i] = idx < BM ? smem_a[tk][idx] : 0;
//      }
//      for (int i = 0; i < TN; i++) {
//        int idx = TN * threadIdx.x + i;
//        reg_b[i] = idx < BN ? smem_b[tk][idx] : 0;
//      }
//      for (int i = 0; i < TM; i++) {
//        for (int j = 0; j < TN; j++) {
//          reg_c[i][j] += reg_a[i] * reg_b[j];
//        }
//      }
//    }
//    __syncthreads();
//  }
//  for (int i = 0; i < TM; i++) {
//      for (int j = 0; j < TN; j++) {
//        int row_c = blockIdx.y * BM + threadIdx.y * TM + i;
//        int col_c = blockIdx.x * BN + threadIdx.x * TN + j;
//       	int idx = row_c * N + col_c;
//        assert(idx < M * N);
//        C[idx] = C[idx] * beta + reg_c[i][j] * alpha;
//      }
//    }
//}