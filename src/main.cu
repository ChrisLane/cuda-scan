/**
 * == Personal ==
 * Name: Christopher Lane
 * ID: 1435876
 *
 * == Goals ==
 * Block Scan: Achieved
 * Full Scan: Not Achieved
 * BCAO: Achieved
 *
 * == Times ==
 ∗ Block scan w/o BCAO: 0.00199 s
 ∗ Block scan w/ BCAO:  0.00214 s
 ∗ Full scan w/o BCAO:
 ∗ Full scan w/ BCAO:
 *
 * == Hardware ==
 * CPU: i7-4710MQ
 * GPU: GTX 860M
 *
 * == Comments ==
 * Implementation:
 * - Length of array is padded to the nearest power of 2 and the array is padded with 0s to that length.
 *
 * Performance:
 *
 */

#include <stdio.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

void refScan(int *h_output, int *h_input, const int len) {
    h_output[0] = 0;

    for (int i = 1; i < len; ++i) {
        h_output[i] = h_input[i - 1] + h_output[i - 1];
    }
}

extern __shared__ int temp[]; // allocated on invocation
__global__ void blockScan(int *d_Output, int *d_Input, int len) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_thid = threadIdx.x;
    int offset = 1;
    int intsPerBlock = blockDim.x << 1;
    int intPosInBlock = local_thid << 1;

    if (thid < len) {
        temp[intPosInBlock] = d_Input[thid << 1]; // load input into shared memory
        temp[intPosInBlock + 1] = d_Input[(thid << 1) + 1];
    } else {
        temp[intPosInBlock] = 0; // load input into shared memory
        temp[intPosInBlock + 1] = 0;
    }

    for (int d = blockDim.x; d > 0; d >>= 1) { // build sum in place up the tree
        __syncthreads();
        if (local_thid < d) {
            int ai = offset * (intPosInBlock + 1) - 1;
            int bi = offset * (intPosInBlock + 2) - 1;

            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    if (local_thid == 0) {
        temp[intsPerBlock - 1] = 0; // clear the last element
    }

    for (int d = 1; d < intsPerBlock; d <<= 1) { // traverse down tree & build scan
        offset >>= 1;
        __syncthreads();
        if (local_thid < d) {
            int ai = offset * (intPosInBlock + 1) - 1;
            int bi = offset * (intPosInBlock + 2) - 1;

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    d_Output[thid << 1] = temp[intPosInBlock]; // write results to device memory
    d_Output[(thid << 1) + 1] = temp[intPosInBlock + 1];
}

__global__ void blockScanNoConflict(int *d_Output, int *d_Input, int len) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_thid = threadIdx.x;
    int offset = 1;
    int intsPerBlock = blockDim.x << 1;
    int intPosInBlock = local_thid << 1;

    int ai = local_thid;
    int bi = local_thid + (blockDim.x);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    if (thid < len) {
        temp[ai + bankOffsetA] = d_Input[thid];
        temp[bi + bankOffsetB] = d_Input[thid + blockDim.x];
    } else {
        temp[ai + bankOffsetA] = 0;
        temp[bi + bankOffsetB] = 0;
    }

    for (int d = blockDim.x; d > 0; d >>= 1) { // build sum in place up the tree
        __syncthreads();
        if (local_thid < d) {
            ai = offset * (intPosInBlock + 1) - 1;
            bi = offset * (intPosInBlock + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    if (local_thid == 0) {
        temp[intsPerBlock - 1 + CONFLICT_FREE_OFFSET(intsPerBlock - 1)] = 0;
    }

    for (int d = 1; d < intsPerBlock; d <<= 1) { // traverse down tree & build scan
        offset >>= 1;
        __syncthreads();
        if (local_thid < d) {
            ai = offset * (intPosInBlock + 1) - 1;
            bi = offset * (intPosInBlock + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    d_Output[ai] = temp[ai + bankOffsetA];
    d_Output[bi] = temp[bi + bankOffsetB];
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

void printTestBlockEquals(int *complete, int *blockOnly, int len, int blockSize) {
    bool equal = true;
    for (int i = 0; i < len; i++) {
        if (complete[i % (blockSize * 2)] != blockOnly[i]) {
            equal = false;
        }
    }

    printf("Test: %s\n", equal ? "PASS" : "FAIL");
}

int main() {
    int len = 10000000; // Number of elements in the input array.
    int blockSize = 128;
    int gridSize = (len + (blockSize * 2) - 1) / (blockSize * 2);
    int sharedMemSize = 2 * blockSize * sizeof(int);
    int testCount = 5;

    double timerResult;

    int *h_Input;
    int *h_Output;
    int *h_Output_d;
    int *d_Input;
    int *d_Output;
    StopWatchInterface *timer = NULL;


    // Allocate host memory
    printf("Allocating host memory...\n");
    h_Input = (int *) malloc(len * sizeof(int));
    h_Output = (int *) malloc(len * sizeof(int));
    h_Output_d = (int *) malloc(len * sizeof(int));

    // Create timer
    sdkCreateTimer(&timer);

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

    // Run reference scan
    refScan(h_Output, h_Input, len);

    // Run Single block scan
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);
    for (int i = 0; i < testCount; i++) {
        blockScan << < gridSize, blockSize, sharedMemSize >> > (d_Output, d_Input, len);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);

    checkCudaErrors(cudaMemcpy(h_Output_d, d_Output, len * sizeof(int), cudaMemcpyDeviceToHost));
    printf("\nBlock Scan Result:\n");
    printf("Grid Size: %d, Block Size: %d\n", gridSize, blockSize);
    printf("Block Scan Result:\n");
    printTestBlockEquals(h_Output, h_Output_d, len, blockSize);
    timerResult = 1.0e-3 * sdkGetTimerValue(&timer) / testCount;
    printf("Time taken: %.5f s, Number of Elements: %d\n\n", timerResult, len);

    // Run block scan no bank conflicts
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);
    for (int i = 0; i < testCount; i++) {
        blockScanNoConflict << < gridSize, blockSize, sharedMemSize >> > (d_Output, d_Input, len);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);

    checkCudaErrors(cudaMemcpy(h_Output_d, d_Output, len * sizeof(int), cudaMemcpyDeviceToHost));
    printf("\nBlock Scan With BCAO Result:\n");
    printf("Grid Size: %d, Block Size: %d\n", gridSize, blockSize);
    printTestBlockEquals(h_Output, h_Output_d, len, blockSize);
    timerResult = 1.0e-3 * sdkGetTimerValue(&timer) / testCount;
    printf("Time taken: %.5f s, Number of Elements: %d\n\n", timerResult, len);

    // Clean up memory
    printf("Cleaning up memory...\n");
    free(h_Input);
    free(h_Output_d);
    checkCudaErrors(cudaFree(d_Output));
    checkCudaErrors(cudaFree(d_Input));
}