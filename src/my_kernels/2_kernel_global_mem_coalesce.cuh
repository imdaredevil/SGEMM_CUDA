#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  int row = blockIdx.y * BLOCKSIZE + threadIdx.x / BLOCKSIZE;
  int col = blockIdx.x * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

  if((row < M) && (col < N)) {
    float sum = 0;
    for(int k=0;k<K;k++) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
  }
}