#include "../utils/image.h"

__global__ void RobertsCrossKernel(IMAGE* in, IMAGE* out)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(x >= in->width - 1)  return;
        if(y >= in->height - 1) return;

        stbi_uc gray    = RGBToGrayscale(GetPixel(in, x,     y    ));
        stbi_uc graySW  = RGBToGrayscale(GetPixel(in, x + 1, y + 1));
        stbi_uc grayS   = RGBToGrayscale(GetPixel(in, x,     y + 1));
        stbi_uc grayW   = RGBToGrayscale(GetPixel(in, x + 1, y    ));

        int gx = gray - graySW;
        int gy = grayS - grayW;

        int mag = abs(gx) + abs(gy);
        if (mag > 255) mag = 255;

        SetPixel(out, x, y, GrayscaleToRGB((stbi_uc)mag));
}

IMAGE* RobertsCross(IMAGE* input)
{
    IMAGE* d_in = CudaImageMalloc(input->width, input->height);
    CudaImageCopy(d_in, input, cudaMemcpyHostToDevice);

    IMAGE* d_out = CudaImageMalloc(input->width, input->height);

    RobertsCrossKernel<<< GetGridDim(input), imgBlockDim >>>(d_in, d_out);
    cudaDeviceSynchronize();

    IMAGE* output = MallocImage(input->width, input->height);
    CudaImageCopy(output, d_out, cudaMemcpyDeviceToHost);

    CudaImageFree(d_in);
    CudaImageFree(d_out);
    return output;
}

