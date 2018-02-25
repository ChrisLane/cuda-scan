/**
 * == Personal ==
 * Name: Christopher Lane
 * ID: 1435876
 *
 * == Goals ==
 * Block Scan: Achieved
 * Full Scan: Achieved
 * BCAO: Achieved
 *
 * == Times ==
 ∗ Block scan w/o BCAO: 2.039ms
 ∗ Block scan w/ BCAO:  1.947ms
 ∗ Full scan w/o BCAO:  27.864ms
 ∗ Full scan w/ BCAO:   27.538ms
 *
 * == Hardware ==
 * CPU: i5-6500
 * GPU: GTX 960
 *
 * == Comments ==
 * Implementation:
 * - Thread results are set to 0 if longer than the array length since the threads will run anyway.
 *   This means that we don't have to pad the input.
 * - Code on the GPU Gems page was wrong in many places. Fixed the code to use blockDim.x in place of len /2 etc.
 *   to make the code work with many grid and block sizes.
 *
 * Performance:
 * - Bitshifting where possible, this would probably be done by the compiler anyway.
 * - CUDA code is compiled with the -O3 flag
 * - Implemented bank conflict avoidance optimisation
 * - A blocksize of 128 gave the fastest results
 *
 */

#include <stdio.h>
#include <time.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) \
    (((n) >> LOG_NUM_BANKS) + ((n) >> (2 * LOG_NUM_BANKS)))
#define NO_CONFLICT true

void refScan(int *h_output, int *h_input, const int len) {
    h_output[0] = 0;

    for (int i = 1; i < len; ++i) {
        h_output[i] = h_input[i - 1] + h_output[i - 1];
    }
}

extern __shared__ int temp[]; // allocated on invocation

__global__ void prescan(int *d_Output, int *d_Input, int len, int *d_SumOutput) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_thid = threadIdx.x;
    int offset = 1;
    int intsPerBlock = blockDim.x << 1;
    int intPosInArray = thid << 1;
    int intPosInBlock = local_thid << 1;

    if (intPosInArray < len) {
        temp[intPosInBlock] = d_Input[intPosInArray]; // load input into shared memory
    } else {
        temp[intPosInBlock] = 0; // load input into shared memory
    }
    if (intPosInArray + 1 < len) {
        temp[intPosInBlock + 1] = d_Input[intPosInArray + 1];
    } else {
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
        if (d_SumOutput) {
            d_SumOutput[blockIdx.x] = temp[intsPerBlock - 1];
        }
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

    d_Output[intPosInArray] = temp[intPosInBlock]; // write results to device memory
    d_Output[intPosInArray + 1] = temp[intPosInBlock + 1];
}

__global__ void prescanNoConflict(int *d_Output, int *d_Input, int len, int *d_SumOutput) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_thid = threadIdx.x;
    int offset = 1;
    int intsPerBlock = blockDim.x << 1;
    int intPosInArray = thid << 1;
    int intPosInBlock = local_thid << 1;

    int g_ai = intPosInBlock;
    int g_bi = intPosInBlock + 1;
    int bankOffsetA = CONFLICT_FREE_OFFSET(g_ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(g_bi);

    if (intPosInArray < len) {
        temp[g_ai + bankOffsetA] = d_Input[intPosInArray];
    } else {
        temp[g_ai + bankOffsetA] = 0;
    }
    if (intPosInArray + 1 < len) {
        temp[g_bi + bankOffsetB] = d_Input[intPosInArray + 1];
    } else {
        temp[g_bi + bankOffsetB] = 0;
    }

    for (int d = blockDim.x; d > 0; d >>= 1) { // build sum in place up the tree
        __syncthreads();
        if (local_thid < d) {
            int ai = offset * (intPosInBlock + 1) - 1;
            int bi = offset * (intPosInBlock + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    if (local_thid == 0) {
        if (d_SumOutput) {
            d_SumOutput[blockIdx.x] = temp[intsPerBlock - 1 + CONFLICT_FREE_OFFSET(intsPerBlock - 1)];
        }
        temp[intsPerBlock - 1 + CONFLICT_FREE_OFFSET(intsPerBlock - 1)] = 0;
    }

    for (int d = 1; d < intsPerBlock; d <<= 1) { // traverse down tree & build scan
        offset >>= 1;
        __syncthreads();
        if (local_thid < d) {
            int ai = offset * (intPosInBlock + 1) - 1;
            int bi = offset * (intPosInBlock + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    if (intPosInArray < len) {
        d_Output[intPosInArray] = temp[g_ai + bankOffsetA];
    }
    if (intPosInArray + 1 < len) {
        d_Output[intPosInArray + 1] = temp[g_bi + bankOffsetB];
    }
}

__global__ void addToBlocks(int *d_Output, int *d_Addition, int len) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if ((thid) < len) {
        d_Output[thid] += d_Addition[blockIdx.x];
    }
}

void fullscan(int *d_Output, int *d_Input, int len, int blockSize, int *d_Sums1Output, int *d_Sums2Output,
              int *d_Sums1Scanned, int *d_Sums2Scanned) {
    int gridSize = (len + (blockSize * 2) - 1) / (blockSize * 2);
    int sharedMemSize = 4 * blockSize * sizeof(int);

    int *h_Output_d = (int *) malloc(len * sizeof(int));
    int *h_SumsOutput_d = (int *) malloc(gridSize * sizeof(int));

    // Run level 1 block scan (Block Scan & Extract Sum1)
#if NO_CONFLICT
    prescanNoConflict <<< gridSize, blockSize, sharedMemSize >>> (d_Output, d_Input, len, d_Sums1Output);
#else
    prescan <<< gridSize, blockSize, sharedMemSize >>> (d_Output, d_Input, len, d_Sums1Output);
#endif
    checkCudaErrors(cudaMemcpy(h_Output_d, d_Output, len * sizeof(int), cudaMemcpyDeviceToHost));


    // Run a block scan on the sum outputs (Block Scan Sum1 & Extract Sum2)
    int sums1Len = gridSize;
    checkCudaErrors(cudaMemcpy(h_SumsOutput_d, d_Sums1Output, sums1Len * sizeof(int), cudaMemcpyDeviceToHost));
    gridSize = (sums1Len + (blockSize * 2) - 1) / (blockSize * 2);
#if NO_CONFLICT
    prescanNoConflict <<< gridSize, blockSize, sharedMemSize >>>
                                                (d_Sums1Scanned, d_Sums1Output, sums1Len, d_Sums2Output);
#else
    prescan <<< gridSize, blockSize, sharedMemSize >>> (d_Sums1Scanned, d_Sums1Output, sums1Len, d_Sums2Output);
#endif
    checkCudaErrors(cudaMemcpy(h_SumsOutput_d, d_Sums1Scanned, sums1Len * sizeof(int), cudaMemcpyDeviceToHost));


    // Run a block scan on the sum outputs (Block Scan Sum2)
    int sums2Len = gridSize;
    checkCudaErrors(cudaMemcpy(h_SumsOutput_d, d_Sums2Output, sums2Len * sizeof(int), cudaMemcpyDeviceToHost));
    gridSize = (sums2Len + (blockSize * 2) - 1) / (blockSize * 2);
#if NO_CONFLICT
    prescanNoConflict <<< gridSize, blockSize, sharedMemSize >>> (d_Sums2Scanned, d_Sums2Output, sums2Len, NULL);
#else
    prescan <<< gridSize, blockSize, sharedMemSize >>> (d_Sums2Scanned, d_Sums2Output, sums2Len, NULL);
#endif
    checkCudaErrors(cudaMemcpy(h_SumsOutput_d, d_Sums2Scanned, sums2Len * sizeof(int), cudaMemcpyDeviceToHost));

    // Add the sums to the original block inputs
    gridSize = (sums1Len + (blockSize * 2) - 1) / (blockSize * 2);
    addToBlocks <<< gridSize, blockSize * 2, sharedMemSize >>> (d_Sums1Scanned, d_Sums2Scanned, sums1Len);
    checkCudaErrors(cudaMemcpy(h_SumsOutput_d, d_Sums1Scanned, sums1Len * sizeof(int), cudaMemcpyDeviceToHost));

    // Add the sums to the original block inputs
    gridSize = (len + (blockSize * 2) - 1) / (blockSize * 2);
    addToBlocks <<< gridSize, blockSize * 2, sharedMemSize >>> (d_Output, d_Sums1Scanned, len);
    checkCudaErrors(cudaMemcpy(h_Output_d, d_Output, len * sizeof(int), cudaMemcpyDeviceToHost));
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
    int sharedMemSize = 4 * blockSize * sizeof(int);
    int testCount = 5;

    double timerResult;

    int *h_Input;
    int *h_Output;
    int *h_Output_d;
    int *d_Input;
    int *d_Output;
    int *d_Sums1Output;
    int *d_Sums2Output;
    int *d_Sums1Scanned;
    int *d_Sums2Scanned;
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
    checkCudaErrors(cudaMalloc(&d_Input, len * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Output, len * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Sums1Output, gridSize * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Sums2Output, gridSize * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Sums1Scanned, gridSize * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Sums2Scanned, gridSize * sizeof(int)));

    printf("Copying input to device...\n");
    checkCudaErrors(cudaMemcpy(d_Input, h_Input, len * sizeof(int), cudaMemcpyHostToDevice));


    /**
     * Reference Scan
     */
    refScan(h_Output, h_Input, len);


    /**
     * Test Block Scan
     */

    // Reset memory
    h_Output_d = (int *) memset(h_Output_d, 0, len * sizeof(int));
    checkCudaErrors(cudaMemset(d_Output, 0, len * sizeof(int)));

    // Start timers
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    // Run kernel
    for (int i = 0; i < testCount; i++) {
#if NO_CONFLICT
        prescanNoConflict <<< gridSize, blockSize, sharedMemSize >>> (d_Output, d_Input, len, NULL);
#else
        prescan <<< gridSize, blockSize, sharedMemSize >>> (d_Output, d_Input, len, NULL);
#endif
    }
    checkCudaErrors(cudaDeviceSynchronize());

    // Stop timer
    sdkStopTimer(&timer);

    // Print results
    checkCudaErrors(cudaMemcpy(h_Output_d, d_Output, len * sizeof(int), cudaMemcpyDeviceToHost));
    printf("\nBlock Scan Result:\n");
    printf("Grid Size: %d, Block Size: %d\n", gridSize, blockSize);
    printTestBlockEquals(h_Output, h_Output_d, len, blockSize);
    timerResult = sdkGetTimerValue(&timer) / testCount;
    printf("Time taken: %.5f ms, Number of Elements: %d\n\n", timerResult, len);


    /**
     * Test Full Scan
     */

    // Reset memory
    h_Output_d = (int *) memset(h_Output_d, 0, len * sizeof(int));
    checkCudaErrors(cudaMemset(d_Output, 0, len * sizeof(int)));

    // Start timers
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    // Run full scan
    for (int i = 0; i < testCount; i++) {
        fullscan(d_Output, d_Input, len, blockSize, d_Sums1Output, d_Sums2Output, d_Sums1Scanned, d_Sums2Scanned);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    // Stop timer
    sdkStopTimer(&timer);

    // Print results
    checkCudaErrors(cudaMemcpy(h_Output_d, d_Output, len * sizeof(int), cudaMemcpyDeviceToHost));
    printf("\nFull Scan Result:\n");
    printf("Grid Size: %d, Block Size: %d\n", gridSize, blockSize);
    printTestEquals(h_Output, h_Output_d, len);
    timerResult = sdkGetTimerValue(&timer) / testCount;
    printf("Time taken: %.5f ms, Number of Elements: %d\n\n", timerResult, len);

    // Clean up memory
    printf("Cleaning up memory...\n");
    free(h_Input);
    free(h_Output_d);
    checkCudaErrors(cudaFree(d_Output));
    checkCudaErrors(cudaFree(d_Input));
}