#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>

#include "image.h"

// This implemententation uses Jorge Jimenez et al.'s improved MLAA algorithm.
// Details can be found at https://www.iryoku.com/mlaa/

IMAGE* MLAA(IMAGE* input);

//In the first pass, edges are detected in the image
IMAGE* DetectEdges(IMAGE* input);
__global__ void DetectEdgesKernel(IMAGE* in, IMAGE* out);

//In the second pass, the blending weights for each pixel adjacent to the edges being smoothed are calculated.
IMAGE* GetBlendWeights(IMAGE* input);
__global__ void GetBlendWeightsKernel(IMAGE* in, IMAGE* out);

//In the third pass, the blending weights are used to blend each pixel with its 4-neighborhood.
IMAGE* AntiAlias(IMAGE* input);
__global__ void AntiAliasKernel(IMAGE* in, IMAGE* out);


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

IMAGE* DetectEdges(IMAGE* input)
{
    IMAGE* d_in = CudaImageMalloc(input->width, input->height);
    CudaImageCopy(d_in, input, cudaMemcpyHostToDevice);

    IMAGE* d_out = CudaImageMalloc(input->width, input->height);

    DetectEdgesKernel<<< GetGridDim(input), imgBlockDim >>>(d_in, d_out);

    IMAGE* output = MallocImage(input->width, input->height);
    CudaImageCopy(output, d_out, cudaMemcpyDeviceToHost);

    CudaImageFree(d_in);
    CudaImageFree(d_out);

    return output;
}

__global__ void DetectEdgesKernel(IMAGE* in, IMAGE* out)
{
    const float edgeLumThreshold = 0.1;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x > in->width) return;
    if(y > in->height) return;

    //Pixels out of image bounds are assumed to have the same color
    PIXEL current = GetPixel(in, x, y);
    PIXEL left = (x - 1 < 0) ? current : GetPixel(in, x - 1, y    );
    PIXEL top  = (y - 1 < 0) ? current : GetPixel(in, x    , y - 1);

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
