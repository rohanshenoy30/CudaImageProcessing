#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "utils/image.h"
#include "algs/upscale.h"
#include "mlaa.h"

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        printf("Specify a path to an image.\n");
        return 1;
    }

    IMAGE* img = LoadImage(argv[1]);
    if(img->data == NULL)
    {
        printf("Error loading the given image.\n");

        free(img);
        return 1;
    }
    printf("Successfully loaded image.\n");

    FreeImage(img);
    return 0;
}
