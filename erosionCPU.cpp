#include <math.h>

#include <limits>

#define imax(a,b) (a > b) ? a : b;
#define imin(a,b) (a < b) ? a : b;

void erosionCPU(int * src, int * dst, int width, int height, int radio) {
    int * tmp = new int[width * height];
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int start_j = imax(0, j - radio);
            int end_j = imin(width - 1, j + radio);
            int value = std::numeric_limits<int>::max();
            for (int jj = start_j; jj <= end_j; jj++) {
                value = imin(src[i * width + jj], value);
            }
            tmp[i * width + j] = value;
        }
    }
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int start_i = imax(0, i - radio);
            int end_i = imin(height - 1, i + radio);
            int value = std::numeric_limits<int>::max();
            for (int ii = start_i; ii <= end_i; ii++) {
                value = imin(tmp[ii * width + j], value);
            }
            dst[i * width + j] = value;
        }
    }
    delete[](tmp);
}