/**
 * == Personal ==
 * Name: Christopher Lane
 * ID: 1435876
 *
 * == Goals ==
 * Block Scan: Not Achieved
 * Full Scan: Not Achieved
 * BCAO: Not Achieved
 *
 * == Times ==
 ∗ Block scan w/o BCAO:
 ∗ Block scan w/ BCAO:
 ∗ Full scan w/o BCAO:
 ∗ Full scan w/ BCAO:
 *
 * == Hardware ==
 * CPU: i7-4710MQ
 * GPU: GTX 860M
 *
 * == Comments ==
 * Implementation:
 *
 * Performance:
 *
 */

// Fix CUDA in CLion
#ifdef __JETBRAINS_IDE__
#define __CUDACC__ 1
#define __host__
#define __device__
#define __global__
#define __forceinline__
#define __shared__

inline void __syncthreads() {}

inline void __threadfence_block() {}

template<class T>
inline T __clz(const T val) { return val; }

struct __cuda_fake_struct {
    int x;
};
#endif


#include <stdio.h>
#include <time.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define NUM_BANKS 32

void refScan(int *h_output, int *h_input, const int len) {
    h_output[0] = 0;

    for (int i = 1; i < len; ++i) {
        h_output[i] = h_input[i - 1] + h_output[i - 1];
    }
}

extern __shared__ int temp[]; // allocated on invocation
__global__ void blockScan(int *d_Output, int *d_Input, const int len) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = 1;

    if (2 * thid < len) {
        temp[2 * thid] = d_Input[2 * thid]; // load input into shared memory
    }
    if (2 * thid + 1 < len) {
        temp[2 * thid + 1] = d_Input[2 * thid + 1];
    }

    for (int d = len >> 1; d > 0; d >>= 1) { // build sum in place up the tree
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;

            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (thid == 0) {
        temp[len - 1] = 0; // clear the last element
    }

    for (int d = 1; d < len; d *= 2) { // traverse down tree & build scan
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    if (2 * thid < len) {
        d_Output[2 * thid] = temp[2 * thid]; // write results to device memory
    }
    if (2 * thid + 1 < len) {
        d_Output[2 * thid + 1] = temp[2 * thid + 1];
    }
}

void printTestEquals(int *value1, int *value2, int len) {
    bool equal = true;
    for (int i = 0; i < len; i++) {
        if (value1[i] != value2[i]) {
            equal = false;
        }
    }

    printf("Test: %s\n", equal ? "PASS" : "FAIL");
}

int main() {
    int blockSize;
    int minGridSize;
    int gridSize;

    int *h_Input;
    int *h_Output;
    int *h_Output_d;
    int *d_Input;
    int *d_Output;
    const int len = 2048; // Number of elements in the input array.

    // Allocate host memory
    printf("Allocating host memory...\n");
    h_Input = (int *) malloc(len * sizeof(int));
    h_Output = (int *) malloc(len * sizeof(int));
    h_Output_d = (int *) malloc(len * sizeof(int));

    // Generate random integers to create input
    printf("Generating random input...\n");
    srand((uint) (time(NULL)));
    for (int i = 0; i < len; i++) {
        h_Input[i] = 1; //rand() % 10;
    }

    // Allocate device memory and copy input to device
    printf("Allocating device memory...\n");
    checkCudaErrors(cudaMalloc((void **) &d_Input, len * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &d_Output, len * sizeof(int)));
    printf("Copying input to device...\n");
    checkCudaErrors(cudaMemcpy(d_Input, h_Input, len * sizeof(int), cudaMemcpyHostToDevice));

    // Set Grid and Block sizes
    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, blockScan, 0, 0));
    gridSize = (len + (blockSize * 2) - 1) / (blockSize * 2);

    // Run Single block scan
    printf("Grid: %d, Block %d\n", gridSize, blockSize);
    blockScan << < gridSize, blockSize, (2 * blockSize) * sizeof(int) >> > (d_Output, d_Input, len);
    checkCudaErrors(cudaDeviceSynchronize());

    refScan(h_Output, h_Input, len);
    /*printf("\nReference Result:\n");
    for (int i = 0; i < len; i++) {
        printf("%d ", h_Output[i]);
    }
    printf("\n");*/

    checkCudaErrors(cudaMemcpy(h_Output_d, d_Output, len * sizeof(int), cudaMemcpyDeviceToHost));
    /*printf("Block Scan Result:\n");
    for (int i = 0; i < len; i++) {
        printf("%d ", h_Output_d[i]);
    }
    printf("\n");*/
    printTestEquals(h_Output, h_Output_d, len);

    // Clean up memory
    printf("Cleaning up memory...\n");
    free(h_Input);
    free(h_Output_d);
    checkCudaErrors(cudaFree(d_Output));
    checkCudaErrors(cudaFree(d_Input));
}