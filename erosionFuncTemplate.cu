
typedef int(*pointFunction_t)(int, int);

__device__ inline int pComputeMin(int a, int b) {
    return (a<b) ? a : b;
}

__device__ inline int pComputeMax(int a, int b) {
    return (a>b) ? a : b;
}

template<const int radio, const pointFunction_t pPointOperation> __device__ void FilterStep2K(int * src, int * dst, int width, int height, int tile_w, int tile_h) {
    extern __shared__ int smem[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int x = bx * tile_w + tx;
    int y = by * tile_h + ty - radio;
    smem[ty * blockDim.x + tx] = 255;
    __syncthreads();
    if (x >= width || y < 0 || y >= height) {
        return;
    }
    smem[ty * blockDim.x + tx] = src[y * width + x];
    __syncthreads();
    if (y < (by * tile_h) || y >= ((by + 1) * tile_h)) {
        return;
    }
    int * smem_thread = &smem[(ty - radio) * blockDim.x + tx];
    int val = smem_thread[0];
#pragma unroll
    for (int yy = 1; yy <= 2 * radio; yy++) {
        val = pPointOperation(val, smem_thread[yy * blockDim.x]);
    }
    dst[y * width + x] = val;
}

template<const int radio, const pointFunction_t pPointOperation> __device__ void FilterStep1K(int * src, int * dst, int width, int height, int tile_w, int tile_h) {
    extern __shared__ int smem[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int x = bx * tile_w + tx - radio;
    int y = by * tile_h + ty;
    smem[ty * blockDim.x + tx] = 255;
    __syncthreads();
    if (x < 0 || x >= width || y >= height) {
        return;
    }
    smem[ty * blockDim.x + tx] = src[y * width + x];
    __syncthreads();
    if (x < (bx * tile_w) || x >= ((bx + 1) * tile_w)) {
        return;
    }
    int * smem_thread = &smem[ty * blockDim.x + tx - radio];
    int val = smem_thread[0];
#pragma unroll
    for (int xx = 1; xx <= 2 * radio; xx++) {
        val = pPointOperation(val, smem_thread[xx]);
    }
    dst[y * width + x] = val;
}

template<const int radio> __global__ void FilterStep1(int * src, int * dst, int width, int height, int tile_w, int tile_h) {
    FilterStep1K<radio, pComputeMin>(src, dst, width, height, tile_w, tile_h);
}

template<const int radio> __global__ void FilterStep2(int * src, int * dst, int width, int height, int tile_w, int tile_h) {
    FilterStep2K<radio, pComputeMin>(src, dst, width, height, tile_w, tile_h);
}

void Filter(int * src, int * dst, int * temp, int width, int height, int radio) {
    // //the host-side function pointer to your __device__ function
    // pointFunction_t h_pointFunction;

    // //in host code: copy the function pointers to their host equivalent
    // cudaMemcpyFromSymbol(&h_pointFunction, pComputeMin, sizeof(pointFunction_t));

    int tile_w1 = 256, tile_h1 = 1;
    dim3 block2(tile_w1 + (2 * radio), tile_h1);
    dim3 grid2(ceil((float)width / tile_w1), ceil((float)height / tile_h1));
    int tile_w2 = 4, tile_h2 = 64;
    dim3 block3(tile_w2, tile_h2 + (2 * radio));
    dim3 grid3(ceil((float)width / tile_w2), ceil((float)height / tile_h2));
    switch (radio) {
        case 1:
            FilterStep1<1><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterStep2<1><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 2:
            FilterStep1<2><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterStep2<2><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 3:
            FilterStep1<3><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterStep2<3><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 4:
            FilterStep1<4><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterStep2<4><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 5:
            FilterStep1<5><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterStep2<5><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 6:
            FilterStep1<6><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterStep2<6><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 7:
            FilterStep1<7><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterStep2<7><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 8:
            FilterStep1<8><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterStep2<8><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 9:
            FilterStep1<9><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterStep2<9><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 10:
            FilterStep1<10><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterStep2<10><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 11:
            FilterStep1<11><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterStep2<11><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 12:
            FilterStep1<12><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterStep2<12><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 13:
            FilterStep1<13><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterStep2<13><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 14:
            FilterStep1<14><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterStep2<14><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 15:
            FilterStep1<15><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterStep2<15><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
    }
    cudaError_t cudaerr = cudaDeviceSynchronize();
}

template<const int radio> __global__ void FilterDStep1(int * src, int * dst, int width, int height, int tile_w, int tile_h) {
    FilterStep1K<radio, pComputeMax>(src, dst, width, height, tile_w, tile_h);
}

template<const int radio> __global__ void FilterDStep2(int * src, int * dst, int width, int height, int tile_w, int tile_h) {
    FilterStep2K<radio, pComputeMax>(src, dst, width, height, tile_w, tile_h);
}

void FilterDilation(int * src, int * dst, int * temp, int width, int height, int radio) {
    // //the host-side function pointer to your __device__ function
    // pointFunction_t h_pointFunction;

    // //in host code: copy the function pointers to their host equivalent
    // cudaMemcpyFromSymbol(&h_pointFunction, pComputeMin, sizeof(pointFunction_t));

    int tile_w1 = 256, tile_h1 = 1;
    dim3 block2(tile_w1 + (2 * radio), tile_h1);
    dim3 grid2(ceil((float)width / tile_w1), ceil((float)height / tile_h1));
    int tile_w2 = 4, tile_h2 = 64;
    dim3 block3(tile_w2, tile_h2 + (2 * radio));
    dim3 grid3(ceil((float)width / tile_w2), ceil((float)height / tile_h2));
    switch (radio) {
        case 1:
            FilterDStep1<1><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterDStep2<1><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 2:
            FilterDStep1<2><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterDStep2<2><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 3:
            FilterDStep1<3><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterDStep2<3><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 4:
            FilterDStep1<4><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterDStep2<4><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 5:
            FilterDStep1<5><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterDStep2<5><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 6:
            FilterDStep1<6><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterDStep2<6><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 7:
            FilterDStep1<7><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterDStep2<7><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 8:
            FilterDStep1<8><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterDStep2<8><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 9:
            FilterDStep1<9><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterDStep2<9><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 10:
            FilterDStep1<10><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterDStep2<10><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 11:
            FilterDStep1<11><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterDStep2<11><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 12:
            FilterDStep1<12><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterDStep2<12><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 13:
            FilterDStep1<13><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterDStep2<13><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 14:
            FilterDStep1<14><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterDStep2<14><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
        case 15:
            FilterDStep1<15><<<grid2,block2,block2.y*block2.x*sizeof(int)>>>(src, temp, width, height, tile_w1, tile_h1);
            (cudaDeviceSynchronize());
            FilterDStep2<15><<<grid3,block3,block3.y*block3.x*sizeof(int)>>>(temp, dst, width, height, tile_w2, tile_h2);
            break;
    }
    cudaError_t cudaerr = cudaDeviceSynchronize();
}