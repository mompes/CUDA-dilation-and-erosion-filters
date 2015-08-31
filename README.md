# Dilation and Erosion filters in CUDA

Several implementations of the dilation and erosion filters are shown:

* CPU:
  1. Using separable filters. ([erosionCPU.cpp](erosionCPU.cpp))
* GPU:  
  2. Naïve implementation, one thread loading n^2 elements per each pixel. ([erosion.cu](erosion.cu))
  3. Separable filter implementation, the processing is divided in two steps and only 2*n elements are loaded per each pixel. ([erosion.cu](erosion.cu))
  4. Shared memory implementation, a tiling approach is used. ([erosion.cu](erosion.cu))
  5. The radio of the filter is templatized to enable unrolling of the main loop. ([erosion.cu](erosion.cu))
  6. The filtering operation is templatized to reuse the same code for the erosion and the dilation. ([erosionFuncTemplate.cu](erosionFuncTemplate.cu))

## Performance

I have performed some tests on a Nvidia GTX 760.

With an image of 1280x1024 and a radio ranging from 2 to 15:

| Radio / Implementation | Speed-up | CPU | Naïve | Separable | Shared mem. | Radio templatized | Filter op. templatized |
| ---------------------- | -------- | --- | ----- | --------- | ----------- | ----------------- | ---------------------- |
| 2 | 34x | 0.07057s | 0.00263s | 0.00213s | 0.00209s | 0.00207s | 0.00207s |
| 3 | 42x | 0.08821s | 0.00357s | 0.00229s | 0.00213s | 0.00211s | 0.00210s |
| 4 | 48x | 0.10283s | 0.00465s | 0.00240s | 0.00213s | 0.00221s | 0.00213s |
| 5 | 56x | 0.12405s | 0.00604s | 0.00258s | 0.00219s | 0.00219s | 0.00221s |
| 10 | 85x | 0.20183s | 0.01663s | 0.00335s | 0.00234s | 0.00237s | 0.00237s |
| 15 | 95x | 0.26114s | 0.03373s | 0.00433s | 0.00287s | 0.00273s | 0.00274s |
