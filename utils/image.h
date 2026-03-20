#pragma once

#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image/stb_image_write.h"

//Image handling
typedef struct 
{
    stbi_uc* data;
    int width;
    int height;
} IMAGE;

const dim3 imgBlockDim  = {32, 32, 1};

IMAGE* MallocImage(int width, int height)
{
    IMAGE* img = (IMAGE*)malloc(sizeof(IMAGE));
    img->data = (stbi_uc*)calloc(width * height, STBI_rgb_alpha);
    img->width = width;
    img->height = height;

    return img;
}

IMAGE* LoadImage(const char* path)
{
    int channels;

    IMAGE* image = (IMAGE*)malloc(sizeof(IMAGE));
    image->data = stbi_load(path, &(image->width), &(image->height), &channels, STBI_rgb_alpha);

    return image;
}

void WriteImage(const char* path, IMAGE* image)
{
    stbi_write_png(path, image->width, image->height, STBI_rgb_alpha, image->data, image->width * STBI_rgb_alpha);
}

void FreeImage(IMAGE* image)
{
    stbi_image_free(image->data);
    free(image);
}

IMAGE* CudaImageMalloc(int width, int height)
{
    int dataSize = width * height * STBI_rgb_alpha;

    stbi_uc* d_data;
    cudaMalloc((void**)&d_data, dataSize);

    IMAGE* d_image;
    cudaMalloc((void**)&d_image, sizeof(IMAGE));

    IMAGE ref = {d_data, width, height};
    cudaMemcpy(d_image, &ref, sizeof(IMAGE), cudaMemcpyHostToDevice);

    return d_image;
}

void CudaImageCopy(IMAGE* dest, IMAGE* src, cudaMemcpyKind kind)
{
    if(kind == cudaMemcpyHostToDevice)
    {
        int dataSize = src->width * src->height * STBI_rgb_alpha;

        IMAGE ref;
        cudaMemcpy(&ref, dest, sizeof(IMAGE), cudaMemcpyDeviceToHost);

        cudaMemcpy(ref.data, src->data, dataSize, cudaMemcpyHostToDevice);
        ref.width = src->width;
        ref.height = src->height;

        cudaMemcpy(dest, &ref, sizeof(IMAGE), cudaMemcpyHostToDevice);
    }
    else if(kind == cudaMemcpyDeviceToHost)
    {
        IMAGE ref;
        cudaMemcpy(&ref, src, sizeof(IMAGE), cudaMemcpyDeviceToHost);

        int dataSize = ref.width * ref.height * STBI_rgb_alpha;

        cudaMemcpy(dest->data, ref.data, dataSize, cudaMemcpyDeviceToHost);
        dest->width = ref.width;
        dest->height = ref.height;
    }
}

void CudaImageFree(IMAGE* image)
{
    IMAGE temp;
    cudaMemcpy(&temp, image, sizeof(IMAGE), cudaMemcpyDeviceToHost);

    cudaFree(temp.data);
    cudaFree(image);
}

dim3 GetGridDim(IMAGE* image)
{
    return 
    { 
        (unsigned int)ceil( image->width   / (float)imgBlockDim.x ), 
        (unsigned int)ceil( image->height  / (float)imgBlockDim.y ), 
        1 
    };
}

//Pixel handling
typedef struct
{
    stbi_uc r;  //Red
    stbi_uc g;  //Green
    stbi_uc b;  //Blue
    stbi_uc a;  //Alpha (transparency)
} PIXEL;

__host__ __device__ int IsPixelInBounds(IMAGE* image, int x, int y)
{
    return 
        x >= 0              || 
        x < image->width    || 
        y >= 0              || 
        y < image->height
    ;
}

__host__ __device__ void ClampToImageBounds(IMAGE* image, int* x, int* y)
{
    if(*x < 0) *x = 0;
    if(*y < 0) *y = 0;

    if(*x >= image->width) *x = image->width - 1;
    if(*y >= image->height) *y = image->height - 1;
}

__host__ __device__ PIXEL GetPixel(IMAGE* image, int x, int y)
{
    ClampToImageBounds(image, &x, &y);

    stbi_uc* pixelAddr = image->data + STBI_rgb_alpha * (image->width * y + x);
    return 
    { 
        pixelAddr[0],       //r
        pixelAddr[1],       //g
        pixelAddr[2],       //b
        pixelAddr[3]        //a
    };
}

__host__ __device__ void SetPixel(IMAGE* image, int x, int y, PIXEL pixel)
{
    if(!IsPixelInBounds(image, x, y)) 
    {
        return;
    }

    stbi_uc* pixelAddr = image->data + STBI_rgb_alpha * (image->width * y + x);
    pixelAddr[0] = pixel.r;
    pixelAddr[1] = pixel.g;
    pixelAddr[2] = pixel.b;
    pixelAddr[3] = pixel.a;
}

