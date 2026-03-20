#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <stdio.h>
#include <errno.h>

#include "../utils/image.h"
#include "../utils/interp.h"

#define AREA_TEX_PATH "area/AreaTex.png"
#define MAX_SEARCH_STEPS 4

// This implemententation uses Jorge Jimenez et al.'s improved MLAA algorithm.
// Details can be found at https://www.iryoku.com/mlaa/

//In the first pass, edges are detected in the image
void DetectEdges(IMAGE* input, IMAGE* output);
__global__ void DetectEdgesKernel(IMAGE* in, IMAGE* out);

//In the second pass, the blending weights for each pixel adjacent to the edges being smoothed are calculated.
void GetBlendWeights(IMAGE* input, IMAGE* output, IMAGE* areas);
__global__ void GetBlendWeightsKernel(IMAGE* in, IMAGE* out, IMAGE* areas);

//In the third pass, the blending weights are used to blend each pixel with its 4-neighborhood.
void BlendNeighborhood(IMAGE* input, IMAGE* weights, IMAGE* output);
__global__ void BlendNeighborhoodKernel(IMAGE* in, IMAGE* weights, IMAGE* out);

IMAGE* LoadAreaTex()
{
    IMAGE* h_areaTex = LoadImage(AREA_TEX_PATH);

    IMAGE* d_areaTex = CudaImageMalloc(h_areaTex->width, h_areaTex->height);
    CudaImageCopy(d_areaTex, h_areaTex, cudaMemcpyHostToDevice);

    FreeImage(h_areaTex);

    return d_areaTex;
}

void MLAA(IMAGE* input, IMAGE* edges, IMAGE* weights, IMAGE* output)
{
    IMAGE* areaTex = LoadAreaTex();

    DetectEdges(input, edges);
    printf("MLAA: 1st pass complete\n");
    GetBlendWeights(edges, weights, areaTex);
    printf("MLAA: 2nd pass complete\n");
    BlendNeighborhood(input, weights, output);
    printf("MLAA: 3rd pass complete\n");

    CudaImageFree(areaTex);
}


/* ============================================ PASS 1 ============================================ */

//uses the CIE XYZ relative luminance (approximate) formula
__device__ float Luminance(PIXEL p)
{
    const float r_factor = 0.2126;
    const float g_factor = 0.7152;
    const float b_factor = 0.0722;
        
    float r_lin = pow(p.r / 255.0, 2.2);
    float g_lin = pow(p.g / 255.0, 2.2);
    float b_lin = pow(p.b / 255.0, 2.2);

    return r_factor * r_lin + g_factor * g_lin + b_factor * b_lin;
}

void DetectEdges(IMAGE* input, IMAGE* output)
{
    IMAGE* d_in = CudaImageMalloc(input->width, input->height);
    CudaImageCopy(d_in, input, cudaMemcpyHostToDevice);

    IMAGE* d_out = CudaImageMalloc(input->width, input->height);

    DetectEdgesKernel<<< GetGridDim(input), imgBlockDim >>>(d_in, d_out);
    cudaDeviceSynchronize();

    CudaImageCopy(output, d_out, cudaMemcpyDeviceToHost);

    CudaImageFree(d_in);
    CudaImageFree(d_out);
}

__global__ void DetectEdgesKernel(IMAGE* in, IMAGE* out)
{
    const float edgeLumThreshold = 0.1;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= in->width)  return;
    if(y >= in->height) return;

    PIXEL current = GetPixel(in, x, y);
    PIXEL left = GetPixel(in, x - 1, y);
    PIXEL top = GetPixel(in, x, y - 1);

    //Luminance is used to compare pixels
    float currentLum = Luminance(current);
    float leftLum = Luminance(left);
    float topLum = Luminance(top);

    //Since two adjacent pixels share an edge, we only check for the top and left edges
    int leftEdge = (abs(currentLum - leftLum) > edgeLumThreshold);
    int topEdge =  (abs(currentLum - topLum ) > edgeLumThreshold);

    //The corresponding pixel in the edge texture
    PIXEL edgel = 
    {
        (stbi_uc)(leftEdge ? 255 : 0),  //Red for if an edge exists on the left
        (stbi_uc)(topEdge  ? 255 : 0),  //Green for if an edge exists on the top
        0,                              //If an edge exists on both sides, the pixel is yellow
        255
    };
    SetPixel(out, x, y, edgel);
}

/* ============================================ PASS 2 ============================================ */

__device__ int SearchXLeft(IMAGE* image, int x, int y)
{
    float xs = x - 1.5;
    float e = 0;

    for(int i = 0; i < MAX_SEARCH_STEPS; i++)               //-ve x
    {
        e = ImageInterp(image, xs, y).g / 255.0;   //normalize to 0..1
        if(e < 0.9)
        {
            return Round(2 * i + 2 * e);
        }

        xs -= 2.0;
    }

    return MAX_SEARCH_STEPS * 2;
}

__device__ int SearchXRight(IMAGE* image, int x, int y)     //+ve x
{
    float xs = x + 1.5;
    float e = 0;

    for(int i = 0; i < MAX_SEARCH_STEPS; i++)
    {
        e = ImageInterp(image, xs, y).g / 255.0;   //normalize to 0..1
        if(e < 0.9)
        {
            return Round(2 * i + 2 * e);
        }

        xs += 2.0;
    }

    return MAX_SEARCH_STEPS * 2;
}

__device__ int SearchYUp(IMAGE* image, int x, int y)        //-ve y
{
    float ys = y - 1.5;
    float e = 0;

    for(int i = 0; i < MAX_SEARCH_STEPS; i++)
    {
        e = ImageInterp(image, x, ys).r / 255.0;   //normalize to 0..1
        if(e < 0.9)
        {
            return Round(2 * i + 2 * e);
        }

        ys -= 2.0;
    }

    return MAX_SEARCH_STEPS * 2;
}

__device__ int SearchYDown(IMAGE* image, int x, int y)      //+ve y
{
    float ys = y + 1.5;
    float e = 0;

    for(int i = 0; i < MAX_SEARCH_STEPS; i++)
    {
        e = ImageInterp(image, x, ys).r / 255.0;   //normalize to 0..1
        if(e < 0.9)
        {
            return Round(2 * i + 2 * e);
        }

        ys += 2.0;
    }

    return MAX_SEARCH_STEPS * 2;
}

__device__ void GetAreas(IMAGE* areaTex, stbi_uc e1, stbi_uc e2, int l, int r, stbi_uc* a1, stbi_uc* a2)
{
    int u = 9 * Round(4 * ((int)e1 / 255.0)) + l;
    int v = 9 * Round(4 * ((int)e2 / 255.0)) + r;

    PIXEL a = GetPixel(areaTex, u, v);
    *a1 = a.r;
    *a2 = a.g;
}

void GetBlendWeights(IMAGE* input, IMAGE* output, IMAGE* areas)
{
    IMAGE* d_in = CudaImageMalloc(input->width, input->height);
    CudaImageCopy(d_in, input, cudaMemcpyHostToDevice);

    IMAGE* d_out = CudaImageMalloc(input->width, input->height);

    GetBlendWeightsKernel<<< GetGridDim(input), imgBlockDim >>>(d_in, d_out, areas);
    cudaDeviceSynchronize();

    CudaImageCopy(output, d_out, cudaMemcpyDeviceToHost);

    CudaImageFree(d_in);
    CudaImageFree(d_out);
}

__global__ void GetBlendWeightsKernel(IMAGE* in, IMAGE* out, IMAGE* areas)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= in->width)  return;
    if(y >= in->height) return;

    PIXEL weights = { 0, 0, 0, 0 };
    PIXEL current = GetPixel(in, x, y);

    if(current.g)   //edge on top
    {
        int l = SearchXLeft(in, x, y);
        int r = SearchXRight(in, x, y);

        stbi_uc e1 = ImageInterp(in, x - l, y - 0.25).r;
        stbi_uc e2 = ImageInterp(in, x + r + 1, y - 0.25).r;

        GetAreas(areas, e1, e2, l, r, &weights.r, &weights.g);
    }
    if(current.r)   //edge on left
    {
        int u = SearchYUp(in, x, y);
        int d = SearchYDown(in, x, y);

        stbi_uc e1 = ImageInterp(in, x - 0.25, y - u).g;
        stbi_uc e2 = ImageInterp(in, x - 0.25, y + d + 1).g;

        GetAreas(areas, e1, e2, u, d, &weights.b, &weights.a);
    }

    SetPixel(out, x, y, weights);
}

/* ============================================ PASS 3 ============================================ */

void BlendNeighborhood(IMAGE* input, IMAGE* weights, IMAGE* output)
{
    IMAGE* d_in = CudaImageMalloc(input->width, input->height);
    CudaImageCopy(d_in, input, cudaMemcpyHostToDevice);

    IMAGE* d_weights = CudaImageMalloc(weights->width, weights->height);
    CudaImageCopy(d_weights, weights, cudaMemcpyHostToDevice);

    IMAGE* d_out = CudaImageMalloc(input->width, input->height);

    BlendNeighborhoodKernel<<< GetGridDim(input), imgBlockDim >>>(d_in, d_weights, d_out);
    cudaDeviceSynchronize();

    CudaImageCopy(output, d_out, cudaMemcpyDeviceToHost);

    CudaImageFree(d_out);
    CudaImageFree(d_weights);
    CudaImageFree(d_in);
}

__global__ void BlendNeighborhoodKernel(IMAGE* in, IMAGE* weights, IMAGE* out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= in->width)  return;
    if(y >= in->height) return;

    PIXEL blends = GetPixel(weights, x, y);
    blends.g = (y >= in->height - 1) ? 0 : GetPixel(weights, x,     y + 1).g;
    blends.a = (x >= in->width  - 1) ? 0 : GetPixel(weights, x + 1, y    ).a;

    float rBlend = blends.r / 255.0;
    float gBlend = blends.g / 255.0;
    float bBlend = blends.b / 255.0;
    float aBlend = blends.a / 255.0;

    float sum = rBlend + gBlend + bBlend + aBlend;
    if(sum > 0)
    {
        PIXEL up    = ImageInterp(in, x,          y - rBlend);
        PIXEL down  = ImageInterp(in, x,          y + gBlend);
        PIXEL left  = ImageInterp(in, x - bBlend, y         );
        PIXEL right = ImageInterp(in, x + aBlend, y         );

        float rResult = (up.r / 255.0) * rBlend + (down.r / 255.0) * gBlend + (left.r / 255.0) * bBlend + (right.r / 255.0) * aBlend;
        float bResult = (up.b / 255.0) * rBlend + (down.b / 255.0) * gBlend + (left.b / 255.0) * bBlend + (right.b / 255.0) * aBlend;
        float gResult = (up.g / 255.0) * rBlend + (down.g / 255.0) * gBlend + (left.g / 255.0) * bBlend + (right.g / 255.0) * aBlend;
        float aResult = (up.a / 255.0) * rBlend + (down.a / 255.0) * gBlend + (left.a / 255.0) * bBlend + (right.a / 255.0) * aBlend;

        PIXEL result = 
        {
            (stbi_uc)Round(255 * rResult / sum),
            (stbi_uc)Round(255 * gResult / sum),
            (stbi_uc)Round(255 * bResult / sum),
            (stbi_uc)Round(255 * aResult / sum)
        };

        SetPixel(out, x, y, result);
    }
    else
    {
        SetPixel(out, x, y, GetPixel(in, x, y));
    }
}

