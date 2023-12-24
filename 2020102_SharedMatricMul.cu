#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <iomanip>

#define N 16
#define BLOCK_SIZE 8

using namespace std;

void PrintMatrix(const int* matrix, int rows, int cols){
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            cout << setw(5) << matrix[i*cols+j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void matrix_multiply(int A[N][N], int B[N][N], int C[N][N]){
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            C[i][j] = 0;
            for(int k = 0; k < N; k++){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

__global__ void matrix_multiply_cuda(int* left, int* right, int* res, int dim){
    int i, j;
    int temp = 0;

    __shared__ int left_shared_t [BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int right_shared_t [BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for(int tileNum = 0; tileNum < gridDim.x; tileNum++){
        j = tileNum * BLOCK_SIZE + threadIdx.x;
        i = tileNum * BLOCK_SIZE + threadIdx.y;

        left_shared_t[threadIdx.y][threadIdx.x] = left[row*dim+j];
        right_shared_t[threadIdx.y][threadIdx.x] = right[i*dim+col];

        __syncthreads();

        for(int k = 0; k < BLOCK_SIZE; k++){
            temp += left_shared_t[threadIdx.y][k] * right_shared_t[k][threadIdx.x];
        }

        __syncthreads();
    }

    res[row*dim+col] = temp
}

int main(){
    int A[N][N], B[N][N], C[N][N];
    int *dev_A, *dev_B, *dev_C;

    srand(time(NULL));

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }
    }

    clock_t start_cpu = clock();
    matrix_multiply(A,B,C);
    clock_t end_cpu = clock();
    double cpu_time = ((double)(end_cpu-start_cpu))/CLOCKS_PER_SEC;

    cout << "Matrix A: " << endl;
    PrintMatrix(&A[0][0], N, N);
    cout << "Matrix B: " << endl;
    PrintMatrix(&B[0][0], N, N);
    cout << "Resultant C: " << endl;
    PrintMatrix(&C[0][0], N, N);

    cudaMalloc((void**)&dev_A, N * N * sizeof(int));
    cudaMalloc((void**)&dev_B, N * N * sizeof(int));
    cudaMalloc((void**)&dev_C, N * N * sizeof(int));

    cudaMemcpy(dev_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice)
    cudaMemcpy(dev_B, B, N * N * sizeof(int), cudaMemcpyHostToDevice)

    dim3 gridSize(N/BLOCK_SIZE, N/BLOCK_SIZE);
    dim3 BlockSize(BLOCK_SIZE, BLOCK_SIZE);

    cudaEvent_t start_gpu, end_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);

    cudaEventRecord(start_gpu);
    matrix_multiply_cuda <<< gridSize, BlockSize >>> (dev_A, dev_B, dev_C, N);
    cudaEventRecord(end_gpu);

    cudaMemcpy(C, dev_C, N * N sizeof(int), cudaMemcpyHostDeviceToHost);

    float gpu_time;
    cudaEventSynchronize(end_gpu);
    cudaEventElapsedTime(&gpu_time, start_gpu, end_gpu);

    cout << "Result from GPU (C_gpu):" << endl;
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