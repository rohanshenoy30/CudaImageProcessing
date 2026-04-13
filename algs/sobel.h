#include "../utils/image.h"

__global__ void SobelKernel(IMAGE* in, IMAGE* out) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= in->width)  return;
    if(y >= in->height) return;

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

    int sumX = 0, sumY = 0;

    for (int i = -1; i <= 1; i++) 
    {
        for (int j = -1; j <= 1; j++) 
        {
            int u = x + i;
            int v = y + j;

            int gray = 0;
            if(IsPixelInBounds(in, u, v))
            {
                PIXEL p = GetPixel(in, x + i, y + j);
                gray = RGBToGrayscale(p);
            }

            sumX += gray * Gx[i + 1][j + 1];
            sumY += gray * Gy[i + 1][j + 1];
        }
    }

    int mag = sqrtf(sumX * sumX + sumY * sumY);
    if(mag > 255) mag = 255;

    SetPixel(out, x, y, GrayscaleToRGB((stbi_uc)mag));
}

IMAGE* Sobel(IMAGE* input)
{
    IMAGE* d_in = CudaImageMalloc(input->width, input->height);
    CudaImageCopy(d_in, input, cudaMemcpyHostToDevice);

    IMAGE* d_out = CudaImageMalloc(input->width, input->height);

    SobelKernel<<< GetGridDim(input), imgBlockDim >>>(d_in, d_out);
    cudaDeviceSynchronize();

    IMAGE* output = MallocImage(input->width, input->height);
    CudaImageCopy(output, d_out, cudaMemcpyDeviceToHost);

    CudaImageFree(d_in);
    CudaImageFree(d_out);
    return output;
}

