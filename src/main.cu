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

__global__ void scanExclusive(int *g_out, int *g_in, int len) {
    extern __shared__ int temp[]; // allocated on invocation
    int threadId = threadIdx.x;
    int pout = 0, pin = 1;
    // Load input into shared memory.
    // This is exclusive scan, so shift right by one
    // and set first element to 0
    temp[pout * len + threadId] = (threadId > 0) ? g_in[threadId - 1] : 0;
    __syncthreads();
    for (int offset = 1; offset < len; offset *= 2) {
        pout = 1 - pout; // swap double buffer indices
        pin = 1 - pout;
        if (threadId >= offset) {
            temp[pout * len + threadId] += temp[pin * len + threadId - offset];
        } else {
            temp[pout * len + threadId] = temp[pin * len + threadId];
        }
        __syncthreads();
    }
    g_out[threadId] = temp[pout * len + threadId]; // write output
}

int main() {
    int *h_Input;
    int *d_Input;
    int *d_Output;
    int length = 100; // Number of elements in the input array.

    // Allocate hardware memory
    h_Input = (int *) malloc(length * sizeof(int));

    // Generate random integers to create input
    srand((uint) (time(NULL)));
    for (int i = 0; i < length; i++) {
        h_Input[i] = rand() % 10;
    }

    // Allocate device memory and copy input to device
    cudaMalloc((void **) &d_Input, length * sizeof(int));
    cudaMalloc((void **) &d_Output, length * sizeof(int));
    cudaMemcpy(d_Input, h_Input, length * sizeof(int), cudaMemcpyHostToDevice);

    printf("%d", h_Input[2]);

    // Clean up memory
    free(h_Input);
    cudaFree(d_Output);
    cudaFree(d_Input);
}