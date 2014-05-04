//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <stdio.h>
//#include "reference_calc.h"

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

#define BLOCK_SIZE 512

__global__
void histogram(const unsigned int * const d_inputVals,
               unsigned int* d_binHistogram,
               const unsigned int mask,
               const unsigned int shift,
               const size_t numElems)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numElems) return;

  unsigned int bin = (d_inputVals[index] & mask) >> shift;
  atomicAdd(&d_binHistogram[bin], 1); 
}

__global__
void scanSerialOnDevice(const unsigned int* const d_binHistogram,
                        unsigned int* d_binScan,
                        const size_t numBins)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index > 0) return; // just one thread will work

  for (unsigned int i = 1; i < numBins; ++i)
  {
    d_binScan[i] = d_binScan[i - 1] + d_binHistogram[i - 1];
  } 
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  const int numBits = 2;
  const int numBins = 1 << numBits;

  unsigned int firstInputValue;
  checkCudaErrors(cudaMemcpy(&firstInputValue, d_inputVals, sizeof(unsigned int), cudaMemcpyDeviceToHost));
  printf("d_inputVals[0] = %u\n", firstInputValue);
  printf("numElems = %u\n", numElems);

  // Create d_binHistogram and d_binScan from d_inputVals
  // First, allocate memory on the GPU.
  unsigned int *d_binHistogram, *d_binScan;
  checkCudaErrors(cudaMalloc((void**)&d_binHistogram, numBins * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void**)&d_binScan, numBins * sizeof(unsigned int)));

  for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i += numBits) 
  {
    unsigned int mask = (numBins - 1) << i;

    // Zero out the bins
    checkCudaErrors(cudaMemset((void*)d_binHistogram, 0, numBins * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(d_binScan, 0, numBins * sizeof(unsigned int)));

    //perform histogram of data & mask into bins
    unsigned int gridSize = (numElems - 1)/BLOCK_SIZE + 1;
    dim3 DimGrid(gridSize, 1, 1);
    dim3 BlockDim(BLOCK_SIZE, 1, 1);

    histogram<<<DimGrid, BlockDim>>>(d_inputVals, d_binHistogram, mask, i, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


    // perform exclusive prefix sum (scan) on binHistogram to get starting
    // location for each bin
    // scan is serial because of small array length
    scanSerialOnDevice<<<1,1>>>(d_binHistogram, d_binScan, numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


    // //Gather everything into the correct location
    // //need to move vals and positions
    // for (unsigned int j = 0; j < numElems; ++j) {
    //   unsigned int bin = (vals_src[j] & mask) >> i;
    //   vals_dst[binScan[bin]] = vals_src[j];
    //   pos_dst[binScan[bin]]  = pos_src[j];
    //   binScan[bin]++;
    // }

    //swap the buffers (pointers only)
    // std::swap(vals_dst, vals_src);
    // std::swap(pos_dst, pos_src);
  }
}
