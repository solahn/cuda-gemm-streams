#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// 행렬 크기 정의
#define N 4096

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

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 타이머용 CUDA 이벤트 생성
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 호스트와 디바이스 메모리 할당
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    h_A = (float*)malloc(size);  // A 행렬 할당
    h_B = (float*)malloc(size);  // B 행렬 할당
    h_C = (float*)malloc(size);  // C 행렬 할당

    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    // 임의의 행렬 데이터 초기화
    for (int i = 0; i < N * N; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        h_C[i] = 0.0f;  // C 초기화
    }

    // 행렬 데이터를 GPU로 복사 (비동기 복사)
    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream));

    // 데이터 복사가 완료된 후 커널 시작되도록 동기화
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 전체 타이머 시작
    CUDA_CHECK(cudaEventRecord(start, 0));

    // GEMM 연산 수행
    gemm(handle, stream, d_A, d_B, d_C, N);

    // 전체 타이머 종료
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // 전체 실행 시간 측정
    float totalTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime, start, stop));

    // 결과를 GPU에서 호스트로 복사 (비동기 복사)
    CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream));

    // 스트림을 동기화하여 결과 복사 완료를 보장
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 결과 출력 (일부 값만 출력)
    std::cout << "Total Execution Time: " << totalTime << " ms" << std::endl;
    std::cout << "GEMM Result (C[0]): " << h_C[0] << std::endl;

    // 자원 해제
    free(h_A); free(h_B); free(h_C);
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    
    // 이벤트 및 스트림 삭제
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}
