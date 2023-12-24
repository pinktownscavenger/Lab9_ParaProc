#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <iomanip>

#define N 500

using namespace std;

void printMatrix(const int *matrix, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::cout << std::setw(5) << matrix[i * cols + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// Sequential matrix multiplication
void matrix_multiply(int A[N][N], int B[N][N], int C[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

__global__ void matrix_multiply_cuda(int *l,int *m, int *n)
{
    int x=blockIdx.x;
    int y=blockIdx.y;
    int k;
  
    n[N*y+x]=0;
    for(k=0;k<N;k++)
    {
      n[N*y+x]=n[N*y+x]+l[N*y+k]*m[N*k+x];
    }
}

int main() {
    int A[N][N], B[N][N], C[N][N];
    int *dev_A, *dev_B, *dev_C;

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }
    }

    clock_t start_cpu = clock();
    matrix_multiply(A, B, C);
    clock_t end_cpu = clock();
    double cpu_time = ((double) (end_cpu - start_cpu)) / CLOCKS_PER_SEC;

    std::cout << "Matrix A:" << std::endl;
    printMatrix(&A[0][0], N, N);
    std::cout << "Matrix B:" << std::endl;
    printMatrix(&B[0][0], N, N);
    std::cout << "Result from CPU (C_cpu):" << std::endl;
    printMatrix(&C[0][0], N, N);

    cudaMalloc((void**)&dev_A, N * N * sizeof(int));
    cudaMalloc((void**)&dev_B, N * N * sizeof(int));
    cudaMalloc((void**)&dev_C, N * N * sizeof(int));

    cudaMemcpy(dev_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 gridSize(N, N);

    cudaEvent_t start_gpu, end_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);

    cudaEventRecord(start_gpu);
    matrix_multiply_cuda<<<gridSize, 1>>>(dev_A, dev_B, dev_C);
    cudaEventRecord(end_gpu);

    cudaMemcpy(C, dev_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    float gpu_time;
    cudaEventSynchronize(end_gpu);
    cudaEventElapsedTime(&gpu_time, start_gpu, end_gpu);

    std::cout << "Result from GPU (C_gpu):" << std::endl;
    printMatrix(&C[0][0], N, N);

    printf("\nMatrix Size %d * %d\n",N,N);
    printf("\nExecution Time:\n");
    printf("CPU Time: %f seconds\n", cpu_time);
    printf("GPU Time: %f milliseconds\n", gpu_time);

    printf("Speed up: %f\n",cpu_time / (gpu_time/1000));

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    return 0;
}
