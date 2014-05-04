//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <stdio.h>
#include <algorithm>
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

  // TODO: put scan here
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

// __global__
// void move_vals_and_positions(const unsigned int* const d_inVals, 
//                              const unsigned int* const d_inPos,
//                              unsigned int* d_outVals,
//                              unsigned int* d_outPos, 
//                              const unsigned int* const d_binScan,
//                              unsigned int mask, 
//                              unsigned int shift, 
//                              const size_t numElems, 
//                              const size_t numBins) 
// {
//    int index = blockIdx.x * blockDim.x + threadIdx.x;
//    int thid = threadIdx.x;
//    if (index >= numElems) return;

//    extern __shared__ unsigned int temp[];

//    int offset = 1;
//    unsigned int bin = (d_inVals[index] & mask) >> shift;
//    for (int i = 0; i < numBins; ++i) 
//    {
//       temp[i*BLOCK_SIZE + thid] = (i == bin) ? 1 : 0; // load input into shared memory
      
//    }

//    if (thid < BLOCK_SIZE/2) 
//    {
//       #pragma UNROLL
//       for (int d = BLOCK_SIZE>>1; d > 0; d >>= 1) // build sum in place up the tree
//       {
//          __syncthreads();
//          if (thid < d)
//          {
//             int ai = offset*(2*thid+1)-1;
//             int bi = offset*(2*thid+2)-1;
//             for (int i = 0; i < numBins; ++i)
//             {
//                temp[i*BLOCK_SIZE + bi] += temp[i*BLOCK_SIZE + ai];
//             }
//          }
//          offset *= 2;
//       }
//       if (thid == 0)  // clear the last element
//       {
//          #pragma UNROLL
//          for (int i = 0; i < numBins; ++i) 
//          {
//             temp[i*BLOCK_SIZE + numBins - 1] = 0;
//          }  
//       } 
      
//       #pragma UNROLL
//       for (int d = 1; d < numBins; d *= 2) // traverse down tree & build scan
//       {
//          offset >>= 1;
//          __syncthreads();
//          if (thid < d)
//          {
//             int ai = offset*(2*thid+1)-1;
//             int bi = offset*(2*thid+2)-1;
//             #pragma UNROLL
//             for (int i = 0; i < numBins; ++i) 
//             {
//                unsigned int t = temp[i*BLOCK_SIZE + ai];
//                temp[i*BLOCK_SIZE + ai] = temp[i*BLOCK_SIZE + bi];
//                temp[i*BLOCK_SIZE + bi] += t;   
//             }
//          }
//       }
//    }
//    __syncthreads();

//    unsigned int pos = temp[bin*BLOCK_SIZE + thid];
//    pos += d_binScan[bin];

//    d_outVals[pos] = d_inVals[index];
//    d_outPos[pos]  = d_inPos[index];
// }

__global__
void gatherInCorrectLocation(const unsigned int* const d_inputVals,
                             const unsigned int* const d_inputPos,
                             unsigned int* d_outputVals,
                             unsigned int* d_outputPos,
                             unsigned int* d_binScan,
                             const unsigned int mask,
                             const unsigned int shift,
                             const size_t numElems)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index > 0) return;

  //Gather everything into the correct location
  //need to move vals and positions
  for (unsigned int j = 0; j < numElems; ++j) 
  {
    unsigned int bin = (d_inputVals[j] & mask) >> shift;
    d_outputVals[d_binScan[bin]] = d_inputVals[j];
    d_outputPos[d_binScan[bin]]  = d_inputPos[j];
    d_binScan[bin]++;
  }
}

void your_sort(unsigned int* const d_inVals,
               unsigned int* const d_inPos,
               unsigned int* const d_outVals,
               unsigned int* const d_outPos,
               const size_t numElems)
{ 
  const int numBits = 2;
  const int numBins = 1 << numBits;

  unsigned int* d_inputVals = d_inVals;
  unsigned int* d_inputPos = d_inPos;
  unsigned int* d_outputVals = d_outVals;
  unsigned int* d_outputPos = d_outPos;

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
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    histogram<<<DimGrid, DimBlock>>>(d_inputVals, d_binHistogram, mask, i, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


    // perform exclusive prefix sum (scan) on binHistogram to get starting
    // location for each bin
    // scan is serial because of small array length
    scanSerialOnDevice<<<1,1>>>(d_binHistogram, d_binScan, numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // Gather everything into the correct location
    // need to move vals and positions
    // move_vals_and_positions<<<DimGrid, DimBlock, BLOCK_SIZE*sizeof(unsigned int)*numBins>>>(
    //                                                               d_inputVals,
    //                                                               d_inputPos,
    //                                                               d_outputVals,
    //                                                               d_outputPos,
    //                                                               d_binScan,
    //                                                               mask,
    //                                                               i,
    //                                                               numElems,
    //                                                               numBins);
    gatherInCorrectLocation<<<1,1>>>(d_inputVals,
                                     d_inputPos,
                                     d_outputVals,
                                     d_outputPos,
                                     d_binScan,
                                     mask,
                                     i,
                                     numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //swap the buffers (pointers only)
    std::swap(d_outputVals, d_inputVals);
    std::swap(d_outputPos, d_inputPos);
  }
  checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
   
  cudaFree(d_binScan);
  cudaFree(d_binHistogram);

  unsigned int* h_outputVals = new unsigned int[numElems];
  checkCudaErrors(cudaMemcpy(h_outputVals, 
                             d_outputVals,
                             numElems * sizeof(unsigned int),
                             cudaMemcpyDeviceToHost));
  for (int i = 100; i < 110; ++i)
  {
    printf("d_outputVals[%d] = %u\n", i, h_outputVals[i]);
  }
}
