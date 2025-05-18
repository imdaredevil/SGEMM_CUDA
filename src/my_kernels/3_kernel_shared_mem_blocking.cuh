#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  __shared__ float shared_A[BLOCKSIZE][BLOCKSIZE];
  __shared__ float shared_B[BLOCKSIZE][BLOCKSIZE];

  int row = blockIdx.y * BLOCKSIZE + threadIdx.x / BLOCKSIZE;
  int col = blockIdx.x * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

  int shared_row = row % BLOCKSIZE;
  int shared_col = col % BLOCKSIZE;
  // int block_start_row = blockIdx.y * BLOCKSIZE;
  // int block_start_col = blockIdx.x * BLOCKSIZE;
                                    
  
  if((row >= M) || (col >= N)) return;
  
  float sum = 0;
  for(int block_start = 0; block_start < K; block_start += BLOCKSIZE) {
    shared_A[shared_row][shared_col] = A[row * K + block_start + shared_col];
    shared_B[shared_row][shared_col] = B[(block_start + shared_row) * N + col];
    __syncthreads();

    for(int k = 0;k < BLOCKSIZE;k++) {
      sum += shared_A[shared_row][k] * shared_B[k][shared_col];
    }
    __syncthreads();
  }

  C[row * N + col] = alpha * sum + beta * C[row * N + col];
  
}