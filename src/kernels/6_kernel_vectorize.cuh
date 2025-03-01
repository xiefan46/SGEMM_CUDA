#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

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


template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmVectorize(int M, int N, int K, float alpha, float *A,
                               float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    // transpose A while loading it
    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      // load C vector into registers
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      // perform GEMM update in reg
      tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
      // write back
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
  }
}