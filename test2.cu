#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// 행렬 크기 정의
#define N 4096
#define BLOCK_SIZE 1024 // 블록 크기 설정

// CUDA 오류 검사 함수
#define CUDA_CHECK(err) if (err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
    exit(EXIT_FAILURE); \
}

// cuBLAS 오류 검사 함수
#define CUBLAS_CHECK(err) if (err != CUBLAS_STATUS_SUCCESS) { \
    std::cerr << "cuBLAS error: " << err << std::endl; \
    exit(EXIT_FAILURE); \
}

// GEMM 연산 함수
void gemm(cublasHandle_t handle, float* A, float* B, float* C, int m, int n, int k) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n, k,
                             &alpha, A, m,
                             B, k,
                             &beta, C, m));
}

int main() {
    // cuBLAS 핸들 및 CUDA 스트림 생성
    cublasHandle_t handle1, handle2;
    CUBLAS_CHECK(cublasCreate(&handle1));
    CUBLAS_CHECK(cublasCreate(&handle2));

    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // 각 핸들에 스트림 설정
    CUBLAS_CHECK(cublasSetStream(handle1, stream1));
    CUBLAS_CHECK(cublasSetStream(handle2, stream2));

    // 타이머용 CUDA 이벤트 생성
    cudaEvent_t start, stop;  // 전체 실행 시간 측정용
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 호스트와 디바이스 메모리 할당
    float *h_A, *h_B, *h_C1, *h_C2;
    float *d_A1, *d_B1, *d_C1;
    float *d_A2, *d_B2, *d_C2;
    size_t size = N * N * sizeof(float);

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C1 = (float*)malloc(size);
    h_C2 = (float*)malloc(size);

    CUDA_CHECK(cudaMalloc(&d_A1, size));
    CUDA_CHECK(cudaMalloc(&d_B1, size));
    CUDA_CHECK(cudaMalloc(&d_C1, size));

    CUDA_CHECK(cudaMalloc(&d_A2, size));
    CUDA_CHECK(cudaMalloc(&d_B2, size));
    CUDA_CHECK(cudaMalloc(&d_C2, size));

    // 임의의 행렬 데이터 초기화 (열 우선 저장)
    for (int i = 0; i < N * N; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 전체 행렬을 GPU로 복사 (비동기 복사)
    CUDA_CHECK(cudaMemcpyAsync(d_A1, h_A, size, cudaMemcpyHostToDevice, stream1));
    CUDA_CHECK(cudaMemcpyAsync(d_B1, h_B, size, cudaMemcpyHostToDevice, stream1));

    CUDA_CHECK(cudaMemcpyAsync(d_A2, h_A, size, cudaMemcpyHostToDevice, stream2));
    CUDA_CHECK(cudaMemcpyAsync(d_B2, h_B, size, cudaMemcpyHostToDevice, stream2));

    // 전체 타이머 시작
    CUDA_CHECK(cudaEventRecord(start));

    // N by N 행렬을 BLOCK_SIZE 단위로 나누어 계산
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        int m = (i + BLOCK_SIZE > N) ? (N - i) : BLOCK_SIZE;
        int k = N;
        int n = N;

        // A, C 포인터 계산 (열 우선 저장)
        float* A_sub1 = d_A1 + i;
        float* C_sub1 = d_C1 + i;
        float* A_sub2 = d_A2 + i;
        float* C_sub2 = d_C2 + i;

        // 각 블록에 대해 GEMM 실행
        gemm(handle1, A_sub1, d_B1, C_sub1, m, n, k);
        gemm(handle2, A_sub2, d_B2, C_sub2, m, n, k);
    }

    // 스트림 동기화
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    // 전체 타이머 종료
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // 전체 실행 시간 측정
    float totalTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime, start, stop));

    // 실행 시간 출력
    std::cout << "Total Execution Time: " << totalTime << " ms" << std::endl;

    // 결과를 GPU에서 호스트로 복사 (비동기 복사)
    CUDA_CHECK(cudaMemcpyAsync(h_C1, d_C1, size, cudaMemcpyDeviceToHost, stream1));
    CUDA_CHECK(cudaMemcpyAsync(h_C2, d_C2, size, cudaMemcpyDeviceToHost, stream2));

    // 스트림을 동기화하여 결과 복사 완료를 보장
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    // 결과 출력 (일부 값만 출력)
    std::cout << "GEMM 1 Result (C1[0]): " << h_C1[0] << std::endl;
    std::cout << "GEMM 2 Result (C2[0]): " << h_C2[0] << std::endl;

    // 자원 해제
    free(h_A); free(h_B); free(h_C1); free(h_C2);
    CUDA_CHECK(cudaFree(d_A1)); CUDA_CHECK(cudaFree(d_B1)); CUDA_CHECK(cudaFree(d_C1));
    CUDA_CHECK(cudaFree(d_A2)); CUDA_CHECK(cudaFree(d_B2)); CUDA_CHECK(cudaFree(d_C2));

    // 이벤트 및 스트림 삭제
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle1));
    CUBLAS_CHECK(cublasDestroy(handle2));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));

    return 0;
}
