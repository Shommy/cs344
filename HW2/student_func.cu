#include "utils.h"

#define BLOCK_WIDTH  16
#define MAX_MASK_WIDTH  9

__constant__ float c_Kernel[MAX_MASK_WIDTH * MAX_MASK_WIDTH];

__global__
void gaussian_blur(const uchar4* const inputChannel,
                   uchar4* const outputChannel,
                   const size_t numRows, const size_t numCols,
                   const int filterWidth)
{
  __shared__  uchar4 buff[BLOCK_WIDTH + MAX_MASK_WIDTH - 1][BLOCK_WIDTH + MAX_MASK_WIDTH - 1];
  int row = blockIdx.y * BLOCK_WIDTH + threadIdx.y;
  int col = blockIdx.x * BLOCK_WIDTH + threadIdx.x;
  int index = row * numCols + col;
  int radius = filterWidth/2;
  int x, y, bx, by;

  if ((row >= numRows) || (col >= numCols)) return;

  // Upper-left corner is loading all apron corners
  if (threadIdx.x < radius && threadIdx.y < radius) {
  // 1. upper-left apron
    x = max(col - radius, 0);
    y = max(row - radius, 0);
    bx = threadIdx.x;
    by = threadIdx.y;
    buff[by][bx] = inputChannel[y * numCols + x];
  
  // 2. upper-right apron
    x = min(col + BLOCK_WIDTH, static_cast<int>(numCols - 1));
    y = max(row - radius, 0);
    bx = (x == numCols - 1) ? (numCols % BLOCK_WIDTH + radius + threadIdx.x) : (radius + BLOCK_WIDTH + threadIdx.x);
    by = threadIdx.y;
    buff[by][bx] = inputChannel[y * numCols + x];

  // 3. lower-left apron
    x = max(col - radius, 0);
    y = min(row + BLOCK_WIDTH, static_cast<int>(numRows - 1));
    bx = threadIdx.x;
    by = (y == numRows - 1) ? (numRows % BLOCK_WIDTH + radius + threadIdx.y) : (radius + BLOCK_WIDTH + threadIdx.y);
    buff[by][bx] = inputChannel[y * numCols + x];

  // 4. lower-right apron
    x = min(col + BLOCK_WIDTH, static_cast<int>(numCols - 1));
    y = min(row + BLOCK_WIDTH, static_cast<int>(numRows - 1));
    bx = (x == numCols - 1) ? (numCols % BLOCK_WIDTH + radius + threadIdx.x) : (radius + BLOCK_WIDTH + threadIdx.x);
    by = (y == numRows - 1) ? (numRows % BLOCK_WIDTH + radius + threadIdx.y) : (radius + BLOCK_WIDTH + threadIdx.y);
    buff[by][bx] = inputChannel[y * numCols + x];
  }

  // Upper radius columns are loading horizontal aprons
  if (threadIdx.y < radius) {
  // 5. upper apron
    x = col;
    y = max(row - radius, 0);
    bx = threadIdx.x + radius;
    by = threadIdx.y;
    buff[by][bx] = inputChannel[y * numCols + x];

  // 6. lower apron
    x = col;
    y = min(row + BLOCK_WIDTH, static_cast<int>(numRows - 1));
    bx = threadIdx.x + radius;
    by = (y == numRows - 1) ? (numRows % BLOCK_WIDTH + radius + threadIdx.y) : (radius + BLOCK_WIDTH + threadIdx.y);
    buff[by][bx] = inputChannel[y * numCols + x];
  }

  // Left radius columns are loading both vertical aprons
  if (threadIdx.x < radius) {
  // 7. left apron
    x = max(col - radius, 0);
    y = row;
    bx = threadIdx.x;
    by = threadIdx.y + radius;
    buff[by][bx] = inputChannel[y * numCols + x];
  
  // 8. right apron
    y = row;
    x = min(col + BLOCK_WIDTH, static_cast<int>(numCols - 1));
    by = radius + threadIdx.y;
    bx = (x == numCols - 1) ? (numCols % BLOCK_WIDTH + radius + threadIdx.x) : (radius + BLOCK_WIDTH + threadIdx.x);
    buff[by][bx] = inputChannel[y * numCols + x];
  }

  // 9. internal (all threads)
  buff[threadIdx.y + radius][threadIdx.x + radius] = inputChannel[index];

  __syncthreads();
  
  float resultRed = 0.f;
  float resultGreen = 0.f;
  float resultBlue = 0.f;
  uchar4 curBuffer;
  float curFilter;
  for(int i = -radius; i <= radius; ++i) {
    for(int j = -radius; j <= radius; ++j) {
      curBuffer = buff[threadIdx.y + i + radius][threadIdx.x + j + radius];
      curFilter = c_Kernel[(i + radius) * filterWidth + j + radius];
      resultRed += curBuffer.x * curFilter;
      resultGreen += curBuffer.y * curFilter;
      resultBlue += curBuffer.z * curFilter;
    }
  }
  unsigned char red = static_cast<unsigned char>(resultRed);
  unsigned char green = static_cast<unsigned char>(resultGreen);
  unsigned char blue = static_cast<unsigned char>(resultBlue);

  outputChannel[index] = make_uchar4(red, green, blue, 255);
}

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{
  int filterSize = filterWidth * filterWidth * sizeof(float);
  checkCudaErrors(cudaMemcpyToSymbol(c_Kernel, h_filter, filterSize));
}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  
  const dim3 blockSize(BLOCK_WIDTH, BLOCK_WIDTH, 1);
  const dim3 gridSize((numCols-1)/BLOCK_WIDTH + 1, (numRows-1)/BLOCK_WIDTH + 1, 1);

  gaussian_blur<<<gridSize, blockSize>>>(d_inputImageRGBA, d_outputImageRGBA, numRows, numCols, filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void cleanup() {
  // Nothing to clean ;)
}
