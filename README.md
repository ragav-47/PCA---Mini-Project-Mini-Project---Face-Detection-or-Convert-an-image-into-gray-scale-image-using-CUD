# Convert Image to Grayscale image using CUDA 

## AIM:

The aim of this project is to demonstrate how to convert an image to grayscale using CUDA programming without relying on the OpenCV library. It serves as an example of GPU-accelerated image processing using CUDA.

## Procedure:
1. Load the input image using the stb_image library.
2. Allocate memory on the GPU for the input and output image buffers.
3. Copy the input image data from the CPU to the GPU.
4. Define a CUDA kernel function that performs the grayscale conversion on each pixel of the image.
5. Launch the CUDA kernel with appropriate grid and block dimensions.
6. Copy the resulting grayscale image data from the GPU back to the CPU.
7. Save the grayscale image using the stb_image_write library.
8. Clean up allocated memory.

## Program:

```cuda c
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <cassert>

#include "stb_image.h"
#include "stb_image_write.h"

struct Pixel
{
    unsigned char r, g, b, a;
};

void ConvertImageToGrayCpu(unsigned char* imageRGBA, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            Pixel* ptrPixel = (Pixel*)&imageRGBA[y * width * 4 + 4 * x];
            unsigned char pixelValue = (unsigned char)(ptrPixel->r * 0.2126f + ptrPixel->g * 0.7152f + ptrPixel->b * 0.0722f);
            ptrPixel->r = pixelValue;
            ptrPixel->g = pixelValue;
            ptrPixel->b = pixelValue;
            ptrPixel->a = 255;
        }
    }
}

__global__ void ConvertImageToGrayGpu(unsigned char* imageRGBA)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t idx = y * blockDim.x * gridDim.x + x;

    Pixel* ptrPixel = (Pixel*)&imageRGBA[idx * 4];
    unsigned char pixelValue = (unsigned char)
        (ptrPixel->r * 0.2126f + ptrPixel->g * 0.7152f + ptrPixel->b * 0.0722f);
    ptrPixel->r = pixelValue;
    ptrPixel->g = pixelValue;
    ptrPixel->b = pixelValue;
    ptrPixel->a = 255;
}

int main(int argc, char** argv)
{
    // Check argument count
    if (argc < 2)
    {
        std::cout << "Usage: 02_ImageToGray <filename>";
        return -1;
    }

    // Open image
    int width, height, componentCount;
    std::cout << "Loading png file...";
    unsigned char* imageData = stbi_load(argv[1], &width, &height, &componentCount, 4);
    if (!imageData)
    {
        std::cout << std::endl << "Failed to open \"" << argv[1] << "\"";
        return -1;
    }
    std::cout << " DONE" << std::endl;

    // Validate image sizes
    if (width % 32 || height % 32)
    {
        // NOTE: Leaked memory of "imageData"
        std::cout << "Width and/or Height is not dividable by 32!";
        return -1;
    }

    
    // Process image on cpu
    std::cout << "Processing image...";
    ConvertImageToGrayCpu(imageData, width, height);
    std::cout << " DONE" << std::endl;
    

    // Copy data to the gpu
    /*
    std::cout << "Copy data to GPU...";
    unsigned char* ptrImageDataGpu = nullptr;
    assert(cudaMalloc(&ptrImageDataGpu, width * height * 4) == cudaSuccess);
    assert(cudaMemcpy(ptrImageDataGpu, imageData, width * height * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    std::cout << " DONE" << std::endl;

    // Process image on gpu
    std::cout << "Running CUDA Kernel...";
    dim3 blockSize(32, 32);
    dim3 gridSize(width / blockSize.x, height / blockSize.y);
    ConvertImageToGrayGpu<<<gridSize, blockSize>>>(ptrImageDataGpu);
    //auto err = cudaGetLastError();
    std::cout << " DONE" << std::endl;

    // Copy data from the gpu
    std::cout << "Copy data from GPU...";
    assert(cudaMemcpy(imageData, ptrImageDataGpu, width * height * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
    std::cout << " DONE" << std::endl;
    */
    // Build output filename
    std::string fileNameOut = argv[1];
    fileNameOut = fileNameOut.substr(0, fileNameOut.find_last_of('.')) + "_gray.png";

    // Write image back to disk
    std::cout << "Writing png to disk...";
    stbi_write_png(fileNameOut.c_str(), width, height, 4, imageData, 4 * width);
    std::cout << " DONE";

    // Free memory
    //cudaFree(ptrImageDataGpu);
    stbi_image_free(imageData);
}
```

## OUTPUT:

### Input Image

![image](https://github.com/curiouzs/PCA---Mini-Project-Mini-Project---Face-Detection-or-Convert-an-image-into-gray-scale-image-using-CUD/assets/75234646/aebcc0ba-a719-41a2-b578-23de1e71ca61)


### Grayscale Image

![image](https://github.com/curiouzs/PCA---Mini-Project-Mini-Project---Face-Detection-or-Convert-an-image-into-gray-scale-image-using-CUD/assets/75234646/fefd6789-7d9f-40e5-9124-e851af7a1084)

## Result:
The CUDA program successfully converts the input image to grayscale using the GPU. The resulting grayscale image is saved as an output file. This example demonstrates the power of GPU parallelism in accelerating image processing tasks.
