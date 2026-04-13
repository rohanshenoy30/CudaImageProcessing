#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void rgb2gray(unsigned char *input, unsigned char *gray, int width, int height, int channels){

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height){
		int idx = (y * width + x) * channels;

		unsigned char r = input[idx + 0];
		unsigned char g = input[idx + 1];
		unsigned char b = input[idx + 2];

		gray[y * width + x] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
	}
}

__global__ void robertsCross(unsigned char *gray, unsigned char *edges, int width, int height){

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width - 1 && y < height - 1){
		int idx = y * width + x;

		int gx = gray[idx] - gray[(y + 1) * width + (x + 1)];
		int gy = gray[(y + 1) * width + x] - gray[y * width + (x + 1)];

		int mag = abs(gx) + abs(gy);

		if (mag > 255) mag = 255;

		edges[idx] = (unsigned char)mag;
	}
}

int main(){
	int width, height, channels;

	unsigned char *h_img = stbi_load("input.png", &width, &height, &channels, 0);
	if (!h_img){
		printf("Failed to load image\n");
		return -1;
	}

	size_t imgSize = width * height * channels;
	size_t graySize = width * height;

	unsigned char *h_edges = (unsigned char *)malloc(graySize);

	unsigned char *d_img, *d_gray, *d_edges;

	cudaMalloc((void **)&d_img, imgSize);
	cudaMalloc((void **)&d_gray, graySize);
	cudaMalloc((void **)&d_edges, graySize);

	cudaMemcpy(d_img, h_img, imgSize, cudaMemcpyHostToDevice);

	dim3 block(16, 16);
	dim3 grid((width + 15) / 16, (height + 15) / 16);

	rgb2gray<<<grid, block>>>(d_img, d_gray, width, height, channels);
	cudaDeviceSynchronize();

	robertsCross<<<grid, block>>>(d_gray, d_edges, width, height);
	cudaDeviceSynchronize();

	cudaMemcpy(h_edges, d_edges, graySize, cudaMemcpyDeviceToHost);

	stbi_write_png("edges.png", width, height, 1, h_edges, width);

	cudaFree(d_img);
	cudaFree(d_gray);
	cudaFree(d_edges);

	stbi_image_free(h_img);
	free(h_edges);

	printf("Done. Saved as edges.png\n");

	return 0;
}
