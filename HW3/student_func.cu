/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Definition Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

//#include "reference_calc.cpp"
#include "utils.h"
#include <float.h>

#define BLOCK_SIZE 1024

__device__ 
inline float dfMax(float x, float y) {
    return (x > y) ? x : y;
}

__device__ 
inline float dfMin(float x, float y) {
    return (x < y) ? x : y;
}

__device__ 
inline void atomicFloatMax(float *address, float value) {
    int oldval, newval, readback;
    oldval = __float_as_int(*address);
    newval = __float_as_int(dfMax(__int_as_float(oldval), value));
    while ((readback=atomicCAS((int *)address, oldval, newval)) != oldval) {
        oldval = readback;
        newval = __float_as_int(dfMax(__int_as_float(oldval), value));
    }
}

__device__ 
inline void atomicFloatMin(float *address, float value) {
    int oldval, newval, readback;
    oldval = __float_as_int(*address);
    newval = __float_as_int(dfMin(__int_as_float(oldval), value));
    while ((readback=atomicCAS((int *)address, oldval, newval)) != oldval) {
        oldval = readback;
        newval = __float_as_int(dfMin(__int_as_float(oldval), value));
    }
}
__global__
void min_max_reduce(const float* const d_inArray,
                    float* d_out,
                    const size_t numElements) 
{
    __shared__ float partialMin[2*BLOCK_SIZE];
    __shared__ float partialMax[2*BLOCK_SIZE];

    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;

    partialMin[t] = (start + t >= numElements) ? FLT_MAX : d_inArray[start+t];
    partialMin[blockDim.x + t] = (start + blockDim.x + t >= numElements) ? FLT_MAX : d_inArray[start + blockDim.x + t];

    partialMax[t] = (start + t >= numElements) ? FLT_MIN : d_inArray[start+t];
    partialMax[blockDim.x + t] = (start + blockDim.x + t >= numElements) ? FLT_MIN : d_inArray[start + blockDim.x + t];

    for(unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            partialMin[t] = min(partialMin[t + stride], partialMin[t]);
            partialMax[t] = max(partialMax[t + stride], partialMax[t]);
        }
    }
    __syncthreads();
    if (t == 0) {
        atomicFloatMax(&d_out[0], partialMax[0]);
    }
    if (t == 1) {
        atomicFloatMin(&d_out[1], partialMin[0]);
    }
}

__global__
void histogram(const float* const d_logLuminance, 
               unsigned int* d_histo, const float range, 
               const float min_lum, const size_t numBins,
               const size_t numElements)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numElements) return;
    unsigned int bin = min(static_cast<unsigned int>(numBins - 1),
                           static_cast<unsigned int>((d_logLuminance[index] - min_lum) / range * numBins));
    atomicAdd(&d_histo[bin], 1);
}

__global__ void exclusive_scan(const unsigned int* const d_histo, unsigned int* const d_cdf, const size_t numBins)
{
    extern __shared__ float temp[]; // allocated on invocation
    int thid = threadIdx.x;
    int offset = 1;
    temp[2*thid] = static_cast<float>(d_histo[2*thid]); // load input into shared memory
    temp[2*thid+1] = static_cast<float>(d_histo[2*thid+1]);
    #pragma UNROLL
    for (int d = numBins>>1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        if (thid < d)
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    if (thid == 0) { temp[numBins - 1] = 0; } // clear the last element
    #pragma UNROLL
    for (int d = 1; d < numBins; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    d_cdf[2*thid] = static_cast<unsigned int>(temp[2*thid]); // write results to device memory
    d_cdf[2*thid+1] = static_cast<unsigned int>(temp[2*thid+1]);
}
void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{    
    size_t numInputElements = numRows * numCols;
    size_t gridSize = (numInputElements-1)/BLOCK_SIZE + 1;
    dim3 DimGrid(gridSize, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);
    
    checkCudaErrors(cudaMemset((float*)d_cdf, FLT_MIN, sizeof(float)));
    checkCudaErrors(cudaMemset((float*)(d_cdf + 1), FLT_MAX, sizeof(float)));

    min_max_reduce<<<DimGrid, DimBlock>>>(d_logLuminance, (float *)d_cdf, numInputElements);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(&max_logLum, (unsigned int*)&d_cdf[0], sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&min_logLum, (unsigned int*)&d_cdf[1], sizeof(float), cudaMemcpyDeviceToHost));

    float range = max_logLum - min_logLum;
    unsigned int *d_histo;
    checkCudaErrors(cudaMalloc(&d_histo, numBins * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(d_histo, 0, numBins * sizeof(unsigned int)));
    
    histogram<<<DimGrid, DimBlock>>>(d_logLuminance, d_histo, range, min_logLum, numBins, numInputElements);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    exclusive_scan<<<1, numBins/2, numBins*sizeof(float)>>>(d_histo, d_cdf, numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaFree(d_histo));

}