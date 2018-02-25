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
 ∗ Block scan w/ BCAO:  1.829ms
 ∗ Full scan w/o BCAO:  3.024ms
 ∗ Full scan w/ BCAO:   2.820ms
 *
 * == Hardware ==
 * CPU: i5-6500
 * GPU: GTX 960
 *
 * == Comments ==
 * Implementation:
 * - Block scans use block ID and dimensions rather than array length in order to work with multiple blocks.
 * - Thread results are set to 0 if longer than the array length since the threads will run anyway.
 *   This means that we don't have to pad the input.
 * - Code on the GPU Gems page was wrong in many places. Fixed the code to use blockDim.x in place of len /2 etc.
 *   to make the code work with many grid and block sizes.
 * - Single level block scans are tested against a known array of 1s requiring more than one block.
 *
 * Performance:
 * - Bitshifting where possible, this would probably be done by the compiler anyway.
 * - CUDA code is compiled with various optimisation flags
 * - Implemented bank conflict avoidance optimisation
 * - A block size of 128 gave the fastest results
 *
 */

#include <stdio.h>
#include <time.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) \
    (((n) >> LOG_NUM_BANKS) + ((n) >> (2 * LOG_NUM_BANKS)))
#define NO_CONFLICT true
#define BLOCK_SIZE 128

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

    temp[intPosInBlock] = intPosInArray < len ? d_Input[intPosInArray] : 0;
    temp[intPosInBlock + 1] = intPosInArray + 1 < len ? d_Input[intPosInArray + 1] : 0;

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

    temp[g_ai + bankOffsetA] = intPosInArray < len ? d_Input[intPosInArray] : 0;
    temp[g_bi + bankOffsetB] = intPosInArray + 1 < len ? d_Input[intPosInArray + 1] : 0;

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

void twoLayer(int *d_Output, int *d_Input, int len, int *d_Sums1Output, int *d_Sums1Scanned) {
    int gridSize = (len + (BLOCK_SIZE * 2) - 1) / (BLOCK_SIZE * 2);
    int level2Grid = (gridSize + (BLOCK_SIZE * 2) - 1) / (BLOCK_SIZE * 2);
    int sharedMemSize = 4 * BLOCK_SIZE * sizeof(int);

    int sums1Len = gridSize;

    int *h_Output_d = (int *) malloc(len * sizeof(int));
    int *h_SumsOutput_d = (int *) malloc(gridSize * sizeof(int));

    // Run level 1 block scan (Block Scan & Extract Sum1)
#if NO_CONFLICT
    prescanNoConflict <<< gridSize, BLOCK_SIZE, sharedMemSize >>> (d_Output, d_Input, len, d_Sums1Output);
#else
    prescan <<< gridSize, BLOCK_SIZE, sharedMemSize >>> (d_Output, d_Input, len, d_Sums1Output);
#endif


    // Run a block scan on the sum outputs (Block Scan Sum1 & Extract Sum2)
#if NO_CONFLICT
    prescanNoConflict <<< level2Grid, BLOCK_SIZE, sharedMemSize >>> (d_Sums1Scanned, d_Sums1Output, sums1Len, NULL);
#else
    prescan <<< level2Grid, BLOCK_SIZE, sharedMemSize >>> (d_Sums1Scanned, d_Sums1Output, sums1Len, NULL);
#endif

    // Add the sums to the original block inputs
    addToBlocks <<< gridSize, BLOCK_SIZE * 2, sharedMemSize >>> (d_Output, d_Sums1Scanned, len);
}

void threeLayer(int *d_Output, int *d_Input, int len, int *d_Sums1Output, int *d_Sums2Output,
                int *d_Sums1Scanned, int *d_Sums2Scanned) {
    int gridSize = (len + (BLOCK_SIZE * 2) - 1) / (BLOCK_SIZE * 2);
    int level2Grid = (gridSize + (BLOCK_SIZE * 2) - 1) / (BLOCK_SIZE * 2);
    int level3Grid = (level2Grid + (BLOCK_SIZE * 2) - 1) / (BLOCK_SIZE * 2);
    int sharedMemSize = 4 * BLOCK_SIZE * sizeof(int);

    int sums1Len = gridSize;
    int sums2Len = level2Grid;

    int *h_Output_d = (int *) malloc(len * sizeof(int));
    int *h_SumsOutput_d = (int *) malloc(gridSize * sizeof(int));

    // Run level 1 block scan (Block Scan & Extract Sum1)
#if NO_CONFLICT
    prescanNoConflict <<< gridSize, BLOCK_SIZE, sharedMemSize >>> (d_Output, d_Input, len, d_Sums1Output);
#else
    prescan <<< gridSize, BLOCK_SIZE, sharedMemSize >>> (d_Output, d_Input, len, d_Sums1Output);
#endif


    // Run a block scan on the sum outputs (Block Scan Sum1 & Extract Sum2)
#if NO_CONFLICT
    prescanNoConflict <<< level2Grid, BLOCK_SIZE, sharedMemSize >>>
                                                (d_Sums1Scanned, d_Sums1Output, sums1Len, d_Sums2Output);
#else
    prescan <<< level2Grid, BLOCK_SIZE, sharedMemSize >>> (d_Sums1Scanned, d_Sums1Output, sums1Len, d_Sums2Output);
#endif


    // Run a block scan on the sum outputs (Block Scan Sum2)
#if NO_CONFLICT
    prescanNoConflict <<< level3Grid, BLOCK_SIZE, sharedMemSize >>> (d_Sums2Scanned, d_Sums2Output, sums2Len, NULL);
#else
    prescan <<< level3Grid, BLOCK_SIZE, sharedMemSize >>> (d_Sums2Scanned, d_Sums2Output, sums2Len, NULL);
#endif

    // Add the sums to the original block inputs
    addToBlocks <<< level2Grid, BLOCK_SIZE * 2, sharedMemSize >>> (d_Sums1Scanned, d_Sums2Scanned, sums1Len);

    // Add the sums to the original block inputs
    addToBlocks <<< gridSize, BLOCK_SIZE * 2, sharedMemSize >>> (d_Output, d_Sums1Scanned, len);
}

void scan(int *d_Output, int *d_Input, int len, int *d_Sums1Output, int *d_Sums2Output,
          int *d_Sums1Scanned, int *d_Sums2Scanned) {
    int gridSize = (len + (BLOCK_SIZE * 2) - 1) / (BLOCK_SIZE * 2);
    int sharedMemSize = 4 * BLOCK_SIZE * sizeof(int);

    if (len <= BLOCK_SIZE * 2) {
#if NO_CONFLICT
        prescanNoConflict <<< gridSize, BLOCK_SIZE, sharedMemSize >>> (d_Output, d_Input, len, NULL);
#else
        prescan <<< gridSize, BLOCK_SIZE, sharedMemSize >>> (d_Output, d_Input, len, NULL);
#endif
    } else if (len <= pow((BLOCK_SIZE * 2), 2)) {
        twoLayer(d_Output, d_Input, len, d_Sums1Output, d_Sums1Scanned);
    } else if (len <= pow((BLOCK_SIZE * 2),3)) {
        threeLayer(d_Output, d_Input, len, d_Sums1Output, d_Sums2Output, d_Sums1Scanned, d_Sums2Scanned);
    } else {
        printf("Array length not within program's abilities.\n");
    }
}

void printTestEquals(int *value1, int *value2, int len) {
    bool equal = true;
    for (int i = 0; i < len; i++) {
        if (value1[i] != value2[i]) {
            equal = false;
            printf("At Index (%d): %d != %d\n", i, value1[i], value2[i]);
            break;
        }
    }

    printf("Test: %s\n", equal ? "PASS" : "FAIL");
}

void printTestBlockEquals(int *complete, int *blockOnly, int len, int blockSize) {
    bool equal = true;
    for (int i = 0; i < len; i++) {
        if (complete[i % (blockSize * 2)] != blockOnly[i]) {
            equal = false;
            printf("At Index (%d): %d != %d\n", i, complete[i % (blockSize * 2)], blockOnly[i]);
            break;
        }
    }

    printf("Test: %s\n", equal ? "PASS" : "FAIL");
}

int main() {
    int len = 10000000; // Number of elements in the input array.
    int gridSize = (len + (BLOCK_SIZE * 2) - 1) / (BLOCK_SIZE * 2);
    int sharedMemSize = 4 * BLOCK_SIZE * sizeof(int);
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
        h_Input[i] = rand() % 10;
    }

    // Allocate device memory and copy input to device
    printf("Allocating device memory...\n");
    checkCudaErrors(cudaMalloc(&d_Input, len * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Output, len * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Sums1Output, gridSize * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Sums2Output, gridSize * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Sums1Scanned, gridSize * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_Sums2Scanned, gridSize * sizeof(int)));


    /**
     * Test Block Scan
     */

    // Generate static input
    for (int i = 0; i < len; i++) {
        h_Input[i] = 1;
    }
    printf("Copying input to device...\n");
    checkCudaErrors(cudaMemcpy(d_Input, h_Input, len * sizeof(int), cudaMemcpyHostToDevice));

    // Run reference scan
    refScan(h_Output, h_Input, len);

    // Reset memory
    h_Output_d = (int *) memset(h_Output_d, 0, len * sizeof(int));
    checkCudaErrors(cudaMemset(d_Output, 0, len * sizeof(int)));

    // Start timers
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    // Run kernel
    for (int i = 0; i < testCount; i++) {
#if NO_CONFLICT
        prescanNoConflict <<< gridSize, BLOCK_SIZE, sharedMemSize >>> (d_Output, d_Input, len, NULL);
#else
        prescan <<< gridSize, BLOCK_SIZE, sharedMemSize >>> (d_Output, d_Input, len, NULL);
#endif
    }
    checkCudaErrors(cudaDeviceSynchronize());

    // Stop timer
    sdkStopTimer(&timer);

    // Print results
    checkCudaErrors(cudaMemcpy(h_Output_d, d_Output, len * sizeof(int), cudaMemcpyDeviceToHost));
    printf("\nBlock Scan Result:\n");
    printf("Grid Size: %d, Block Size: %d\n", gridSize, BLOCK_SIZE);
    printTestBlockEquals(h_Output, h_Output_d, len, BLOCK_SIZE);
    timerResult = sdkGetTimerValue(&timer) / testCount;
    printf("Time taken: %.5f ms, Number of Elements: %d\n\n", timerResult, len);


    /**
     * Test Full Scan
     */

    // Generate random input
    srand((uint) (time(NULL)));
    for (int i = 0; i < len; i++) {
        h_Input[i] = rand() % 10;
    }
    printf("Copying input to device...\n");
    checkCudaErrors(cudaMemcpy(d_Input, h_Input, len * sizeof(int), cudaMemcpyHostToDevice));

    // Run reference scan
    refScan(h_Output, h_Input, len);

    // Reset memory
    h_Output_d = (int *) memset(h_Output_d, 0, len * sizeof(int));
    checkCudaErrors(cudaMemset(d_Output, 0, len * sizeof(int)));

    // Start timers
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    // Run full scan
    for (int i = 0; i < testCount; i++) {
        scan(d_Output, d_Input, len, d_Sums1Output, d_Sums2Output, d_Sums1Scanned, d_Sums2Scanned);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    // Stop timer
    sdkStopTimer(&timer);

    // Print results
    checkCudaErrors(cudaMemcpy(h_Output_d, d_Output, len * sizeof(int), cudaMemcpyDeviceToHost));
    printf("\nGPU Scan Result:\n");
    printTestEquals(h_Output, h_Output_d, len);
    timerResult = sdkGetTimerValue(&timer) / testCount;
    printf("Time taken: %.5f ms, Number of Elements: %d\n\n", timerResult, len);

    // Clean up memory
    printf("Cleaning up memory...\n");
    free(h_Input);
    free(h_Output);
    free(h_Output_d);
    checkCudaErrors(cudaFree(d_Output));
    checkCudaErrors(cudaFree(d_Input));
    checkCudaErrors(cudaFree(d_Sums1Output));
    checkCudaErrors(cudaFree(d_Sums2Output));
    checkCudaErrors(cudaFree(d_Sums1Scanned));
    checkCudaErrors(cudaFree(d_Sums2Scanned));
}