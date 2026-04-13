#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "algs/upscale.h"
#include "algs/mlaa.h"
#include "algs/sobel.h"
#include "algs/cross.h"

#define EXE_NAME "imgproc.exe"

//mkdir commands
#ifdef _WIN32
    #include <direct.h>
#else
    #include <sys/stat.h>
    #include <sys/types.h>
#endif

//commands
const char* helpCommand    = "help";
const char* upscaleCommand = "upscale";
const char* mlaaCommand    = "mlaa";
const char* sobelCommand   = "sobel";
const char* crossCommand   = "cross";

int CompareCommand(char* arg, const char* command)
{
    return strcmp(arg, command) == 0;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        printf("Specify a command. Run \"./%s %s\" to list all available commands.\n", EXE_NAME, helpCommand);
        return 1;
    }

    char* command = argv[1];
    if(CompareCommand(command, helpCommand))
    {
        printf("[./imgproc.exe %s]: Show command list\n", helpCommand);
        printf("[./imgproc.exe %s <filepath> [<factor>]]: Upscale the given image by an optional integer <factor>. Default upscaling factor is 4.\n", upscaleCommand); 
        printf("[./imgproc.exe %s <filepath>]: Antialias image using MLAA\n", mlaaCommand);
        printf("[./imgproc.exe %s <filepath>]: Run Sobel edge detection on the given image\n", sobelCommand);
        printf("[./imgproc.exe %s <filepath>]: Run Roberts cross edge detection on the given image\n", crossCommand);
        return 0;
    }
    
    if(argc < 3)
    {
        printf("Specify a file path.\n");
        return 2;
    }

    char* filepath = argv[2];
    IMAGE* img = LoadImage(filepath);
    if(img->data == NULL)
    {
        printf("Error loading the image at %s\n", filepath);

        free(img);
        return 3;
    }

    //create ./output/ if it doesn't exist
    int status;
    #ifdef _WIN32
        status = _mkdir("output");
    #else
        status = mkdir("output", S_IRWXU | S_IRWXO | S_IRWXG);
    #endif
    if(status != 0 && errno != EEXIST)
    {
        perror("Could not create output directory");

        FreeImage(img);
        return 4;
    }

    if(CompareCommand(command, upscaleCommand))
    {
        int factor = 4;
        if(argc >= 4)
        {
            sscanf(argv[3], "%d", &factor);
            if(factor < 1)
            {
                printf("Non-positive factors are not allowed.\n");

                FreeImage(img);
                return 5;
            }
        }

        IMAGE* upscaled = Upscale(img, factor);
        printf("Image upscaled. Writing to output...\n");

        WriteImage("output/upscaled.png", upscaled);
        printf("Upscaled image written to output.\n");

        FreeImage(upscaled);
    }
    else if(CompareCommand(command, mlaaCommand))
    {
        IMAGE* edges = MallocImage(img->width, img->height);
        IMAGE* weights = MallocImage(img->width, img->height);
        IMAGE* output = MallocImage(img->width, img->height);
        MLAA(img, edges, weights, output);

        printf("MLAA completed. Writing to output...\n");

        WriteImage("output/mlaa_pass1.png", edges);
        WriteImage("output/mlaa_pass2.png", weights);
        WriteImage("output/mlaa_pass3.png", output);
        printf("Images written to output.\n");

        FreeImage(edges);
        FreeImage(weights);
        FreeImage(output);
    }
    else if(CompareCommand(command, sobelCommand))
    {
        IMAGE* sobel = Sobel(img);
        printf("Sobel edge detection complete. Writing to output...\n");

        WriteImage("output/sobel.png", sobel);
        printf("Image written to output.\n");

        FreeImage(sobel);
    }
    else if(CompareCommand(command, crossCommand))
    {
        IMAGE* cross = RobertsCross(img);
        printf("Roberts cross edge detection complete. Writing to output...\n");

        WriteImage("output/cross.png", cross);
        printf("Image written to output.\n");

        FreeImage(cross);
    }
    else
    {
        printf("Command unrecognized. Run \"./%s %s\" to list all available commands.\n", EXE_NAME, helpCommand);
    }

    FreeImage(img);
    return 0;
}

