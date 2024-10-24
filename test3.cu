#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

// CUDA 에러 핸들링 매크로
#define CUDA_CHECK(err) if(err != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
    exit(EXIT_FAILURE); \
}

#define CUBLAS_CHECK(err) if(err != CUBLAS_STATUS_SUCCESS) { \
    std::cerr << "cuBLAS Error: " << err << std::endl; \
    exit(EXIT_FAILURE); \
}

const int N = 4096; // 행렬 크기 (큰 작업)
const int BLOCK_SIZE = 2048; // 작은 작업으로 나눌 크기

// GEMM 수행 함수
void gemm(cublasHandle_t handle, float* A, float* B, float* C, int n, cudaStream_t stream) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasSetStream(handle, stream);
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n, &beta, C, n));
}

int main() {
    // cuBLAS 핸들 생성
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // 두 개의 스트림 생성
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // 큰 행렬 할당 (N x N)
    float *A1, *B1, *C1, *A2, *B2, *C2;
    CUDA_CHECK(cudaMalloc(&A1, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B1, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C1, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&A2, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B2, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C2, N * N * sizeof(float)));

    // 큰 작업 - 실행 시간 측정
    auto start = std::chrono::high_resolution_clock::now();
    gemm(handle, A1, B1, C1, N, stream1);
    gemm(handle, A2, B2, C2, N, stream2);
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Large tasks execution time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
              << " ms" << std::endl;

    // 작은 작업으로 나누기 (BLOCK_SIZE x BLOCK_SIZE)
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            gemm(handle, A1 + i * N + j, B1 + i * N + j, C1 + i * N + j, BLOCK_SIZE, stream1);
            gemm(handle, A2 + i * N + j, B2 + i * N + j, C2 + i * N + j, BLOCK_SIZE, stream2);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Small tasks execution time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
              << " ms" << std::endl;

    // 메모리 해제
    CUDA_CHECK(cudaFree(A1));
    CUDA_CHECK(cudaFree(B1));
    CUDA_CHECK(cudaFree(C1));
    CUDA_CHECK(cudaFree(A2));
    CUDA_CHECK(cudaFree(B2));
    CUDA_CHECK(cudaFree(C2));

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));

    return 0;
}
