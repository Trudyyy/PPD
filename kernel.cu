#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>
using namespace std;
using namespace cv;

int nrOfCols;
int nrOfRows;

cudaError_t addWithCuda(Mat* imgIn1, Mat* imgOut1);

__global__ void kernel(int* red, int* green, int* blue, int* redOut, int* greenOut, int* blueOut, int nrOfRows, int nrOfCols)
{
    /*
    0 -1 0
    -1 5 -1
    0 -1 0
    */
    int j = blockIdx.x;
    int i = threadIdx.x;

    int value[3] = { 0, 0, 0 };    

    int sI = i - 2;
    int sJ = j - 2;

    int fI = i + 2;
    int fJ = j + 2;

    for (int x = sI; x <= fI; x++) {
        for (int y = sJ; y <= fJ; y++) {
            if (x >= 0 && y >= 0 && x < nrOfRows && y < nrOfCols) {
                int index = (nrOfCols)*x + y;
                value[0] = value[0] + red[index];
                value[1] = value[1] + green[index] ;
                value[2] = value[2] + blue[index] ;
                /*
                if (x == i && y == j) {  
                    value[0] = value[0] + red[index] * 5;
                    value[1] = value[1] + green[index] * 5;
                    value[2] = value[2] + blue[index] * 5;
                }
                else if (x == i || y == j) {
                    value[0] = value[0] + red[index] * (-1);
                    value[1] = value[1] + green[index] * (-1);
                    value[2] = value[2] + blue[index] * (-1);
                }*/
            }
        }
    }
    redOut[(nrOfCols) * i + j] = value[0]/25;
    greenOut[(nrOfCols) * i + j] = value[1]/25;
    blueOut[(nrOfCols) * i + j] = value[2]/25;    
}

void writeToFile(Mat* img) {
    ofstream red;
    ofstream green;
    ofstream blue;
    red.open("red.txt");
    green.open("green.txt");
    blue.open("blue.txt");
    for (int i = 0; i < nrOfRows; i++) {
        for (int j = 0; j < nrOfCols; j++) {
            Vec3b color = img->at<Vec3b>(i,j);
            //cout << (int)color.val[0] << endl;
            red << (int)color.val[0] << " ";
            green << (int)color.val[1] << " ";
            blue << (int)color.val[2] << " ";

        }
        red << endl;
        green << endl;
        blue << endl;
    }
    red.close();
    green.close();
    blue.close();
}

int* getVec(ifstream* MyReadFile) {
    int* rez = new int[nrOfRows*nrOfCols+1];
    int r = 0;
    
    string word;

    while (*MyReadFile >> word)
    {
        rez[r] = stoi(word);
        r++;
    }
    MyReadFile->close();
    
    return rez;
}

int startWithoutReading(Mat& img, Mat& imgOut) {
    cudaError_t cudaStatus = addWithCuda(&img, &imgOut);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    namedWindow("Original photo", WINDOW_NORMAL);
    imshow("Original photo", img);
    resizeWindow("Original photo", nrOfCols / 2, nrOfRows / 2);

    namedWindow("Sharpen photo", WINDOW_NORMAL);
    imshow("Sharpen photo", imgOut);
    resizeWindow("Sharpen photo", nrOfCols / 2, nrOfRows / 2);

    int k = waitKey(0); // Wait for a keystroke in the window
    if (k == 's')
    {
        imwrite("out.jpg", imgOut);
    }
}
int start(string fileName) {
    string image_path = samples::findFile(fileName);
    Mat img = imread(image_path, IMREAD_COLOR);

    nrOfCols = img.cols;
    nrOfRows = img.rows;

    Mat imgOut = imread(image_path, IMREAD_COLOR);

    cout << "Dim: "<<nrOfCols << "x" << nrOfRows << endl;

    writeToFile(&img);

    if (img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    startWithoutReading(img,imgOut);
}
int main()
{
    int x;
    cout << "~~~~~~~~~~\n";
    cout << "0.Exit\n";
    cout << "1.Portrait\n";
    cout << "2.Colors\n";
    cout << "3.Pink\n";
    cout << ">>";
    cin >> x;
    while (x != 0) {
        if (x == 1)start("in.jpg");
        else if (x == 2)start("in2.jpg");
        else if (x == 3)start("in3.png");
        cout << "~~~~~~~~~~\n";
        cout << "0.Exit\n";
        cout << "1.Portrait\n";
        cout << "2.Colors\n";
        cout << "3.Pink\n";
        cout << ">>";
        cin >> x;
    }
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(Mat* imgIn, Mat* imgOut)
{
    ifstream redFile("red.txt");
    ifstream greenFile("green.txt");
    ifstream blueFile("blue.txt");
    int* rr = getVec(&redFile);
    int* gg = getVec(&greenFile);
    int* bb = getVec(&blueFile);

    int* red;
    int* green;
    int* blue;
    int* redOut;
    int* greenOut;
    int* blueOut;

    
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&red, nrOfRows  * nrOfCols * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&green, nrOfRows * nrOfCols * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&blue, nrOfRows * nrOfCols * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&redOut, nrOfRows * nrOfCols * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&greenOut, nrOfRows * nrOfCols * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&blueOut, nrOfRows * nrOfCols * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
   
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(red, rr, nrOfRows * nrOfCols * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(green, gg, nrOfRows * nrOfCols * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(blue, bb, nrOfRows * nrOfCols * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    auto start = std::chrono::steady_clock::now();
    // Launch a kernel on the GPU with one thread for each element.
    kernel<<<nrOfCols, nrOfRows>>>(red, green, blue, redOut, greenOut, blueOut, nrOfRows, nrOfCols);
    

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(rr, redOut, nrOfRows * nrOfCols * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(gg, greenOut, nrOfRows * nrOfCols * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(bb, blueOut, nrOfRows * nrOfCols * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    for (int i = 0; i < nrOfRows; i++) {
        for (int j = 0; j < nrOfCols; j++) {
            int index = nrOfCols * i + j;
            Vec3b color = imgOut->at<Vec3b>(i, j);
            color[0] = (rr[index] < 255) ? (rr[index] > 0) ? rr[index] : 0 : 255;
            color[1] = (gg[index] < 255) ? (gg[index] > 0) ? gg[index] : 0 : 255;
            color[2] = (bb[index] < 255) ? (bb[index] > 0) ? bb[index] : 0 : 255;
            imgOut->at<Vec3b>(i, j) = color;
        }
    }
   
Error:
    cudaFree(red);
    cudaFree(green);
    cudaFree(blue);
    cudaFree(redOut);
    cudaFree(greenOut);
    cudaFree(blueOut);
    
    return cudaStatus;
}
