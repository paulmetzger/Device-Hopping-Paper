// TODO: The code into which this template will be pasted might also use T. It doesn't at the moment though.
#define T %TYPE%

// The following class is a workaround for using dynamically sized
// shared memory in templated code. Without this workaround, the
// compiler would generate two shared memory arrays (one for SP
// and one for DP) of the same name and would generate an error.
class SharedMem {
public:
    __device__ inline T* getPointer()
    {
        extern __shared__ T s[];
        return s;
    };
};

__global__ void reduce(const T* __restrict__ g_idata, T* __restrict__ g_odata, const unsigned int n) {
    const unsigned int tid = threadIdx.x;
    unsigned int i = (blockIdx.x*(blockDim.x*2)) + tid;
    const unsigned int gridSize = blockDim.x*2*gridDim.x;

    // Shared memory will be used for intrablock summation
    // NB: CC's < 1.3 seem incompatible with the templated dynamic
    // shared memory workaround.
    // Inspection with cuda-gdb shows sdata as a pointer to *global*
    // memory and incorrect results are obtained.  This explicit macro
    // works around this issue. Further, it is acceptable to specify
    // float, since CC's < 1.3 do not support double precision.
#if __CUDA_ARCH__ <= 130
    extern volatile __shared__ float sdata[];
#else
    SharedMem shared;
    volatile T* sdata = shared.getPointer();
#endif

    sdata[tid] = 0.0f;

    // Reduce multiple elements per thread
    while (i < n)
    {
        T x = 0.0;
        if (i + BLOCK_SIZE_X < n) x = g_idata[i+BLOCK_SIZE_X];
        sdata[tid] += g_idata[i] + x;
        i += gridSize;
    }
    __syncthreads();

    // Reduce the contents of shared memory
    // NB: This is an unrolled loop, and assumes warp-syncrhonous
    // execution.
    if (BLOCK_SIZE_X >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE_X >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE_X >= 128)
    {
        if (tid < 64)  { sdata[tid] += sdata[tid + 64]; }  __syncthreads();
    }
    if (tid < warpSize)
    {
        // NB2: This section would also need __sync calls if warp
        // synchronous execution were not assumed
        if (BLOCK_SIZE_X >= 64) sdata[tid] += sdata[tid + 32];
        if (BLOCK_SIZE_X >= 32) sdata[tid] += sdata[tid + 16];
        if (BLOCK_SIZE_X >= 16) sdata[tid] += sdata[tid + 8];
        if (BLOCK_SIZE_X >= 8)  sdata[tid] += sdata[tid + 4];
        if (BLOCK_SIZE_X >= 4)  sdata[tid] += sdata[tid + 2];
        if (BLOCK_SIZE_X >= 2)  sdata[tid] += sdata[tid + 1];
    }

    // Write result for this block to global memory
    if (tid == 0) g_odata[blockIdx.x] += sdata[0];
}