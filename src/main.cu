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
    int outputAi = ai;
    int outputBi = bi;
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
            outputAi = offset * ((thid << 1) + 1) - 1;
            outputBi = offset * ((thid << 1) + 2) - 1;
            outputAi += CONFLICT_FREE_OFFSET(outputAi);
            outputBi += CONFLICT_FREE_OFFSET(outputBi);

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    d_Output[outputAi] = temp[ai + bankOffsetA];
    d_Output[outputBi] = temp[bi + bankOffsetB];
}

__global__ void level1(int *d_Output, int *d_Input, int len, int *d_SumOutput) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_thid = threadIdx.x;
    int offset = 1;
    int intsPerBlock = blockDim.x << 1;
    int intPosInBlock = local_thid << 1;

    int ai = local_thid;
    int bi = local_thid + (blockDim.x);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    int outputAi = ai;
    int outputBi = bi;
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
        d_SumOutput[blockIdx.x] = temp[intsPerBlock - 1 + CONFLICT_FREE_OFFSET(intsPerBlock - 1)];
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
            outputAi = offset * ((thid << 1) + 1) - 1;
            outputBi = offset * ((thid << 1) + 2) - 1;
            outputAi += CONFLICT_FREE_OFFSET(outputAi);
            outputBi += CONFLICT_FREE_OFFSET(outputBi);

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    d_Output[outputAi] = temp[ai + bankOffsetA];
    d_Output[outputBi] = temp[bi + bankOffsetB];
}

__global__ void addToBlocks(int *d_Output, int *d_IncrOutput) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    d_Output[thid] += d_IncrOutput[blockIdx.x];
}

void fullscan(int *d_Output, int *d_Input, int len, int blockSize) {
    int gridSize = (len + (blockSize * 2) - 1) / (blockSize * 2);
    int sharedMemSize = blockSize * 2;
    int *d_SumsOutput;
    int *d_IncrOutput;
    int *h_SumsOutput_d;
    int *h_IncrOutput_d;
    int *h_Output_d;

    h_Output_d = (int *) malloc(len * sizeof(int));
    h_SumsOutput_d = (int *) malloc(gridSize * sizeof(int));
    h_IncrOutput_d = (int *) malloc(gridSize * sizeof(int));
    h_Output_d = (int *) malloc(len * sizeof(int));

    checkCudaErrors(cudaMalloc((void **) &d_SumsOutput, gridSize * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &d_IncrOutput, gridSize * sizeof(int)));

    level1 << < gridSize, blockSize, sharedMemSize >> > (d_Output, d_Input, len, d_SumsOutput);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_SumsOutput_d, d_SumsOutput, gridSize * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_Output_d, d_Output, len * sizeof(int), cudaMemcpyDeviceToHost));

    printf("Block Scan Result:\n");
    for (int i = 0; i < len; i++) {
        printf("%d ", h_Output_d[i]);
    }
    printf("\n");

    printf("SUM Array:\n");
    for (int i = 0; i < gridSize; i++) {
        printf("%d ", h_SumsOutput_d[i]);
    }
    printf("\n");

    blockScanNoConflict << < gridSize, blockSize * 2, sharedMemSize >> >
                                                      (d_IncrOutput, d_SumsOutput, gridSize * sizeof(int));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_IncrOutput_d, d_IncrOutput, gridSize * sizeof(int), cudaMemcpyDeviceToHost));
    printf("INCR Scan Result:\n");
    for (int i = 0; i < gridSize; i++) {
        printf("%d ", h_IncrOutput_d[i]);
    }
    printf("\n");

    printf("Grid: %d, Block %d\n", gridSize, blockSize);
    addToBlocks << < gridSize, blockSize * 2, sharedMemSize >> > (d_Output, d_IncrOutput);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_Output_d, d_Output, len * sizeof(int), cudaMemcpyDeviceToHost));
    printf("Add to Blocks Result:\n");
    for (int i = 0; i < len; i++) {
        printf("%d ", h_Output_d[i]);
    }
    printf("\n");
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
    int len = 32; // Number of elements in the input array.
    int blockSize = 2;
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


    /**
     * Reference Scan
     */
    refScan(h_Output, h_Input, len);


    /**
     * Test Naive Block Scan
     */

    // Start timers
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    // Run kernel
    for (int i = 0; i < testCount; i++) {
        blockScan << < gridSize, blockSize, sharedMemSize >> > (d_Output, d_Input, len);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    // Stop timer
    sdkStopTimer(&timer);

    // Print results
    checkCudaErrors(cudaMemcpy(h_Output_d, d_Output, len * sizeof(int), cudaMemcpyDeviceToHost));
    printf("\nBlock Scan Result:\n");
    printf("Grid Size: %d, Block Size: %d\n", gridSize, blockSize);
    printf("Block Scan Result:\n");
    printTestBlockEquals(h_Output, h_Output_d, len, blockSize);
    timerResult = 1.0e-3 * sdkGetTimerValue(&timer) / testCount;
    printf("Time taken: %.5f s, Number of Elements: %d\n\n", timerResult, len);


    /**
     * Test Block Scan with Bank Collision Avoidance Optimisation
     */

    // Reset memory
    h_Output_d = (int *) memset(h_Output_d, 0, len * sizeof(int));
    checkCudaErrors(cudaMemset(d_Output, 0, len * sizeof(int)));

    // Start timers
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    // Run kernel
    for (int i = 0; i < testCount; i++) {
        blockScanNoConflict << < gridSize, blockSize, sharedMemSize >> > (d_Output, d_Input, len);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    // Stop timer
    sdkStopTimer(&timer);

    // Print results
    checkCudaErrors(cudaMemcpy(h_Output_d, d_Output, len * sizeof(int), cudaMemcpyDeviceToHost));
    printf("\nBlock Scan With BCAO Result:\n");
    printf("Grid Size: %d, Block Size: %d\n", gridSize, blockSize);
    printTestBlockEquals(h_Output, h_Output_d, len, blockSize);
    timerResult = 1.0e-3 * sdkGetTimerValue(&timer) / testCount;
    printf("Time taken: %.5f s, Number of Elements: %d\n\n", timerResult, len);


    /**
     * Test Full Scan
     */
    h_Output_d = (int *) memset(h_Output_d, 0, len * sizeof(int));
    checkCudaErrors(cudaMemset(d_Output, 0, len * sizeof(int)));
    fullscan(d_Output, d_Input, len, blockSize);

    // Clean up memory
    printf("Cleaning up memory...\n");
    free(h_Input);
    free(h_Output_d);
    checkCudaErrors(cudaFree(d_Output));
    checkCudaErrors(cudaFree(d_Input));
}