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
    __shared__ float smem_a[BM][BK];
    __shared__ float smem_b[BK][BN];
    float reg_c[TM][TN] = {0};
   	if (threadIdx.x == 0 && threadIdx.y == 0) {
    	printf("block cnt: %d , thread cnt: %d \n", gridDim.x * gridDim.y, blockDim.x * blockDim.y);
   	}

    assert(BM % TM == 0);
    assert(BN % TN == 0);
    const int THREAD_CNT_PER_BLOCK = blockDim.x * blockDim.y;
	assert(THREAD_CNT_PER_BLOCK == (BM * BN) / (TM * TN));

	const int ELEMENT_PER_THREAD_A = BM * BK /  THREAD_CNT_PER_BLOCK;
    const int ELEMENT_PER_THREAD_B = BK * BN / THREAD_CNT_PER_BLOCK;
    for (int bk = 0; bk < K; bk += BK) {
        // Load data to smem
        int offset_a = (threadIdx.y * blockDim.x + threadIdx.x) * ELEMENT_PER_THREAD_A;
        assert(offset_a < BM * BK);
        #pragma unroll
        for (int i = 0; i < ELEMENT_PER_THREAD_A; i++) {
          const int offset_a_row = (offset_a + i) / BK;
          const int offset_a_col = (offset_a + i) % BK;
          if (BM * blockIdx.y + offset_a_row < M && bk + offset_a_col < K) {
            smem_a[offset_a_row][offset_a_col] = A[(BM * blockIdx.y + offset_a_row) * K + bk + offset_a_col];
          } else {
            smem_a[offset_a_row][offset_a_col] = 0.0;
          }

        }

        #pragma unroll
        int offset_b = (threadIdx.y * blockDim.x + threadIdx.x) * ELEMENT_PER_THREAD_B;
        assert(offset_b < BN * BK);
        for (int i = 0; i < ELEMENT_PER_THREAD_B; i++) {
          const int offset_b_row = (offset_b + i) / BN;
          const int offset_b_col = (offset_b + i) % BN;
          if (bk + offset_b_row < K && BN * blockIdx.x + offset_b_col < N) {
            smem_b[offset_b_row][offset_b_col] = B[(bk + offset_b_row) * N + BN * blockIdx.x + offset_b_col];
          } else {
            smem_b[offset_b_row][offset_b_col] = 0.0;
          }

        }

        __syncthreads();

        // load data to registers
        float reg_a[TM];
        float reg_b[TN];


        assert(TM * THREAD_CNT_PER_BLOCK == BM * BK);

        for (int tk = 1; tk < bk; tk++) {
			#pragma unroll
        	for (int i = 0; i < TM; i++) {
          		reg_a[i] = threadIdx.y * TM + i < BM ? smem_a[threadIdx.y * TM + i][tk] : 0.0;
        	}
        	#pragma unroll
        	for (int i = 0; i < TN; i++) {
          		reg_b[i] = threadIdx.x * TN + i < BN ? smem_b[tk][threadIdx.x * TN + i] : 0.0;
        	}
        	#pragma unroll
        	for (int i = 0; i < TM; i++) {
          		for (int j = 0; j < TN; j++) {
            		reg_c[i][j] += reg_a[i] * reg_b[j];
          		}
        	}
        	__syncthreads();
        }

    }

    // Write register result to C
    const int c_offset_x = BN * blockIdx.x + threadIdx.x * TN;
    const int c_offset_y = BM * blockIdx.y + threadIdx.y * TM;
    #pragma unroll
    for (int i = 0; i < TM; i++) {
      for (int j = 0; j < TN; j++) {
        const int c_row = c_offset_y + i;
        const int c_col = c_offset_x + j;
        if (c_row < M && c_col < N) {
          const int index = c_row * N + c_col;
          C[index] = C[index] * beta + alpha * reg_c[i][j];
        }

      }
    }
}
