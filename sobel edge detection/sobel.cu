// sobel.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BLOCK_SIZE 16

__global__ void sobelKernel(unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int sumX = 0, sumY = 0;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int pixel = input[(y + i) * width + (x + j)];
                sumX += pixel * Gx[i + 1][j + 1];
                sumY += pixel * Gy[i + 1][j + 1];
            }
        }

        int mag = sqrtf(sumX * sumX + sumY * sumY);
        if (mag > 255) mag = 255;

        output[y * width + x] = (unsigned char)mag;
    }
}

int main() {
    int width, height, channels;

    unsigned char *img = stbi_load("input.png", &width, &height, &channels, 1);

    if (!img) {
        printf("Error loading image\n");
        return -1;
    }

    size_t size = width * height * sizeof(unsigned char);

    unsigned char *d_input, *d_output;
    unsigned char *output = (unsigned char*)malloc(size);

    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    cudaMemcpy(d_input, img, size, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    sobelKernel<<<grid, block>>>(d_input, d_output, width, height);

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    stbi_write_png("output.png", width, height, 1, output, width);

    printf("Done! Output saved as output.png\n");

    cudaFree(d_input);
    cudaFree(d_output);
    stbi_image_free(img);
    free(output);

    return 0;
}