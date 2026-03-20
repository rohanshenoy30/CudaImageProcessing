#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "utils/image.h"
#include "utils/interp.h"
#include "algs/upscale.h"
#include "algs/mlaa.h"

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

    IMAGE* edges = MallocImage(img->width, img->height);
    IMAGE* weights = MallocImage(img->width, img->height);
    IMAGE* output = MallocImage(img->width, img->height);
    MLAA(img, edges, weights, output);
    printf("MLAA completed\n");

    WriteImage("output/mlaa_pass1.png", edges);
    WriteImage("output/mlaa_pass2.png", weights);
    WriteImage("output/mlaa_pass3.png", output);

    FreeImage(edges);
    FreeImage(weights);
    FreeImage(output);
    FreeImage(img);
    return 0;
}

