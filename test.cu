#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// 행렬 크기 정의
#define N 4096
#define BLOCK_SIZE 4096 // 간격

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
void gemm(cublasHandle_t handle, cudaStream_t stream, float* A, float* B, float* C, int n) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 행렬 곱셈: C = alpha * A * B + beta * C
    cublasSetStream(handle, stream);
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                             &alpha, A, n, B, n, &beta, C, n));
}

int main() {
    // cuBLAS 핸들 및 CUDA 스트림 생성
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // 타이머용 CUDA 이벤트 생성
    cudaEvent_t start1, stop1, start2, stop2;
    cudaEvent_t start, stop;  // 전체 실행 시간 측정용
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&stop1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop2));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop)); // 괄호 수정

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

    // 임의의 행렬 데이터 초기화
    for (int i = 0; i < N * N; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 전체 타이머 시작
    CUDA_CHECK(cudaEventRecord(start, 0));

    // N by N 행렬을 BLOCK_SIZE 단위로 나누어 계산
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        int blockSize = (i + BLOCK_SIZE > N) ? (N - i) : BLOCK_SIZE;

        // 행렬 데이터를 GPU로 복사 (비동기 복사)
        CUDA_CHECK(cudaMemcpyAsync(d_A1 + i * N, h_A + i * N, blockSize * N * sizeof(float), cudaMemcpyHostToDevice, stream1));
        CUDA_CHECK(cudaMemcpyAsync(d_B1 + i * N, h_B + i * N, blockSize * N * sizeof(float), cudaMemcpyHostToDevice, stream1));
        CUDA_CHECK(cudaMemcpyAsync(d_A2 + i * N, h_A + i * N, blockSize * N * sizeof(float), cudaMemcpyHostToDevice, stream2));
        CUDA_CHECK(cudaMemcpyAsync(d_B2 + i * N, h_B + i * N, blockSize * N * sizeof(float), cudaMemcpyHostToDevice, stream2));

        // 각 블록에 대해 GEMM 실행 및 시작 시간 기록
        auto startTime1 = std::chrono::high_resolution_clock::now(); // 시스템 시간 기록
        gemm(handle, stream1, d_A1 + i * N, d_B1 + i * N, d_C1 + i * N, blockSize);
        auto endTime1 = std::chrono::high_resolution_clock::now();

        auto startTime2 = std::chrono::high_resolution_clock::now(); // 시스템 시간 기록
        gemm(handle, stream2, d_A2 + i * N, d_B2 + i * N, d_C2 + i * N, blockSize);
        auto endTime2 = std::chrono::high_resolution_clock::now();

        // 시작 시점 출력
        auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(startTime1.time_since_epoch()).count();
        auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(startTime2.time_since_epoch()).count();
        std::cout << "Block starting at row " << i << " - Stream 1 Start Time: " << duration1 << " ms, Stream 2 Start Time: " << duration2 << " ms" << std::endl;

        // 동기화
        CUDA_CHECK(cudaStreamSynchronize(stream1));
        CUDA_CHECK(cudaStreamSynchronize(stream2));
    }

    // 전체 타이머 종료
    CUDA_CHECK(cudaEventRecord(stop, 0));
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
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(stop1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(stop2));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop)); // 괄호 수정
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));

    return 0;
}
