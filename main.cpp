

#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <chrono>
#include <ctime>

#include "erosionFuncTemplate.h"
#include "erosionCPU.h"
#include "erosion.h"

inline int cudaDeviceInit(int argc, const char **argv)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaSetDevice(0);

    return 0;
}

void populateImage(int * image, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            image[i * width + j] = rand() % 256;
        }
    }
}

void diff(int *himage, int *dimage, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (himage[i * width + j] != dimage[i * width + j]) {
                std::cout << "Expected: " << himage[i * width + j] << ", actual: " << dimage[i * width + j] << ", on: " << i << ", " << j << std::endl;
                exit(0);
            }
        }
    }
}

int main(int argc, char *argv[])
{
    cudaDeviceInit(argc, (const char **)argv);

    int * dimage_src, *dimage_dst, *dimage_tmp;
    int * himage_src, *himage_dst, *himage_tmp;
    // Width and height of the image
    int width = 1280, height = 1024, radio = 5;

    (cudaMalloc(&dimage_src, width * height * sizeof(int)));
    (cudaMalloc(&dimage_dst, width * height * sizeof(int)));
    (cudaMalloc(&dimage_tmp, width * height * sizeof(int)));
    (cudaMallocHost(&himage_src, width * height * sizeof(int)));
    (cudaMallocHost(&himage_dst, width * height * sizeof(int)));
    (cudaMallocHost(&himage_tmp, width * height * sizeof(int)));

    // Randomly populate the image
    populateImage(himage_src, width, height);

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
 
    for (radio = 2; radio <= 15; radio++) {
        // Calculate the eroded image on the host
        erosionCPU(himage_src, himage_dst, width, height, radio);

        end = std::chrono::system_clock::now(); 
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::cout << "Erosion CPU: " << elapsed_seconds.count() << "s\n";

        start = std::chrono::system_clock::now();
        // Copy the image from the host to the GPU
        (cudaMemcpy(dimage_src, himage_src, width * height * sizeof(int), cudaMemcpyHostToDevice));
        // Calculate the eroded image on the GPU
        NaiveErosion(dimage_src, dimage_dst, width, height, radio);    
        // Copy the eroded image to the host
        (cudaMemcpy(himage_tmp, dimage_dst, width * height * sizeof(int), cudaMemcpyDeviceToHost));
        end = std::chrono::system_clock::now(); 
        elapsed_seconds = end-start;
        std::cout << "GPU Naive erosion: " << elapsed_seconds.count() << "s\n";
        // Diff the images
        diff(himage_dst, himage_tmp, width, height);

        start = std::chrono::system_clock::now();
        // Copy the image from the host to the GPU
        (cudaMemcpy(dimage_src, himage_src, width * height * sizeof(int), cudaMemcpyHostToDevice));
        ErosionTwoSteps(dimage_src, dimage_dst, dimage_tmp, width, height, radio);
        // Copy the eroded image to the host
        (cudaMemcpy(himage_tmp, dimage_dst, width * height * sizeof(int), cudaMemcpyDeviceToHost));
        end = std::chrono::system_clock::now(); 
        elapsed_seconds = end-start;
        std::cout << "GPU two steps erosion: " << elapsed_seconds.count() << "s\n";
        // Diff the images
        diff(himage_dst, himage_tmp, width, height);

        start = std::chrono::system_clock::now();
        // Copy the image from the host to the GPU
        (cudaMemcpy(dimage_src, himage_src, width * height * sizeof(int), cudaMemcpyHostToDevice));
        ErosionTwoStepsShared(dimage_src, dimage_dst, dimage_tmp, width, height, radio);
        // Copy the eroded image to the host
        (cudaMemcpy(himage_tmp, dimage_dst, width * height * sizeof(int), cudaMemcpyDeviceToHost));
        end = std::chrono::system_clock::now(); 
        elapsed_seconds = end-start;
        std::cout << "GPU two steps shared erosion: " << elapsed_seconds.count() << "s\n";
        // Diff the images
        diff(himage_dst, himage_tmp, width, height);

        start = std::chrono::system_clock::now();
        // Copy the image from the host to the GPU
        (cudaMemcpy(dimage_src, himage_src, width * height * sizeof(int), cudaMemcpyHostToDevice));
        ErosionTemplateSharedTwoSteps(dimage_src, dimage_dst, dimage_tmp, width, height, radio);
        // Copy the eroded image to the host
        (cudaMemcpy(himage_tmp, dimage_dst, width * height * sizeof(int), cudaMemcpyDeviceToHost));
        end = std::chrono::system_clock::now(); 
        elapsed_seconds = end-start;
        std::cout << "GPU two steps shared template erosion: " << elapsed_seconds.count() << "s\n";
        // Diff the images
        diff(himage_dst, himage_tmp, width, height);

        start = std::chrono::system_clock::now();
        // Copy the image from the host to the GPU
        (cudaMemcpy(dimage_src, himage_src, width * height * sizeof(int), cudaMemcpyHostToDevice));
        Filter(dimage_src, dimage_dst, dimage_tmp, width, height, radio);
        // Copy the eroded image to the host
        (cudaMemcpy(himage_tmp, dimage_dst, width * height * sizeof(int), cudaMemcpyDeviceToHost));
        end = std::chrono::system_clock::now(); 
        elapsed_seconds = end-start;
        std::cout << "GPU two steps shared template erosion with a function templated: " << elapsed_seconds.count() << "s\n";
        // Diff the images
        diff(himage_dst, himage_tmp, width, height);

        start = std::chrono::system_clock::now();
        // Copy the image from the host to the GPU
        (cudaMemcpy(dimage_src, himage_src, width * height * sizeof(int), cudaMemcpyHostToDevice));
        FilterDilation(dimage_src, dimage_dst, dimage_tmp, width, height, radio);
        // Copy the eroded image to the host
        (cudaMemcpy(himage_tmp, dimage_dst, width * height * sizeof(int), cudaMemcpyDeviceToHost));
        end = std::chrono::system_clock::now(); 
        elapsed_seconds = end-start;
        std::cout << "GPU two steps shared template dilation with a function templated: " << elapsed_seconds.count() << "s\n";
    }
    //templateErosionTophatCall(dimage_src, dimage_dst, dimage_tmp, width, height, radio);


    std::cout << "Great!!" << std::endl;

    cudaFree(dimage_src);
    cudaFree(dimage_dst);
    cudaFree(dimage_tmp);
    cudaFreeHost(himage_src);
    cudaFreeHost(himage_dst);
    cudaFreeHost(himage_tmp);
    cudaDeviceReset();
    return 0;
}
