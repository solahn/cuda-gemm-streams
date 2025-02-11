#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// 행렬 크기 정의
#define N 2048
#define BLOCK_SIZE 1024  // 블록 크기 정의

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

// GEMM 연산 함수 (단일 블록)
void gemm_block(cublasHandle_t handle, cudaStream_t stream, float* A, float* B, float* C, int n, int block_start, int block_size) {
    // 스트림 설정
    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 서브 행렬에 대한 포인터 계산
    float* A_sub = A + block_start;   // 행 우선 저장이므로 열 방향 이동
    float* C_sub = C + block_start;

    // GEMM 연산 수행
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_T,  // A와 B를 전치
                             n, block_size, n,
                             &alpha,
                             B, n,
                             A_sub, n,
                             &beta,
                             C_sub, n));
}

int main() {
    // 각 스트림마다 별도의 cuBLAS 핸들 생성
    cublasHandle_t handle1, handle2;
    CUBLAS_CHECK(cublasCreate(&handle1));
    CUBLAS_CHECK(cublasCreate(&handle2));

    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // 타이머용 CUDA 이벤트 생성
    cudaEvent_t start, stop;  // 전체 실행 시간 측정용
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 호스트와 디바이스 메모리 할당
    float *h_A, *h_B, *h_C1, *h_C2;
    float *d_A1, *d_B1, *d_C1;
    float *d_A2, *d_B2, *d_C2;
    size_t size = N * N * sizeof(float);

    h_A = (float*)malloc(size);  // 동일한 A 행렬 사용
    h_B = (float*)malloc(size);  // 동일한 B 행렬 사용
    h_C1 = (float*)malloc(size); // C1 결과 저장
    h_C2 = (float*)malloc(size); // C2 결과 저장

    CUDA_CHECK(cudaMalloc(&d_A1, size));
    CUDA_CHECK(cudaMalloc(&d_B1, size));
    CUDA_CHECK(cudaMalloc(&d_C1, size));

    CUDA_CHECK(cudaMalloc(&d_A2, size));
    CUDA_CHECK(cudaMalloc(&d_B2, size));
    CUDA_CHECK(cudaMalloc(&d_C2, size));

    // 임의의 행렬 데이터 초기화 (행 우선 저장)
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            h_A[row * N + col] = static_cast<float>(rand()) / RAND_MAX;
            h_B[row * N + col] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // 행렬 데이터를 GPU로 복사
    CUDA_CHECK(cudaMemcpy(d_A1, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B1, h_B, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A2, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B2, h_B, size, cudaMemcpyHostToDevice));

    // 전체 타이머 시작
    CUDA_CHECK(cudaEventRecord(start, 0));

    // 블록 단위로 GEMM 연산 수행 (같은 for문 내에서 두 스트림 실행)
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        int current_block_size = std::min(BLOCK_SIZE, N - i);

        // 스트림 1에서 블록 처리
        gemm_block(handle1, stream1, d_A1, d_B1, d_C1, N, i, current_block_size);

        // 스트림 2에서 블록 처리
        gemm_block(handle2, stream2, d_A2, d_B2, d_C2, N, i, current_block_size);
    }

    // 두 스트림 동기화
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    // 전체 타이머 종료
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // 전체 실행 시간 측정
    float totalTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime, start, stop));

    // 실행 시간 출력
    std::cout << "Total Execution Time: " << totalTime << " ms" << std::endl;

    // 결과를 GPU에서 호스트로 복사
    CUDA_CHECK(cudaMemcpy(h_C1, d_C1, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C2, d_C2, size, cudaMemcpyDeviceToHost));

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
