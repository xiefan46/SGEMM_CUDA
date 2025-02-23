#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < N && y < M) {
        float sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += A[y * K + k] * B[k * N + x];
        }
        C[y * N + x] = sum * alpha + beta * C[y * N + x];
    }
}

//__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
//                            const float *B, float beta, float *C) {
//    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
//    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
//
//    // if statement is necessary to make things work under tile quantization
//    if (x < M && y < N) {
//        float tmp = 0.0;
//        for (int i = 0; i < K; ++i) {
//            tmp += A[x * K + i] * B[i * N + y];
//        }
//        // C = α*(A@B)+β*C
//        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
//    }
//}