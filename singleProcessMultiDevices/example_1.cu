//
// Example 1: Single Process, Single Thread, Multiple Devices
//

#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <stdlib.h>
#include <vector>
#include <iostream>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

__global__ void  init1(float *dptr,int i)
{
  int id = threadIdx.x;
  dptr[id] = id;
  printf("GPU: %d,dptr: %f\n",i,dptr[id]);
}

int main(int argc, char *argv[])
{
  std::cout << "================================================================"<<
               "\n    Executing " << argv[0] << " now!\n"<<
               "================================================================\n";
  ncclComm_t comms[2];

  // managing 2 devices
  int nDev = 2;
  const int size = 3;

  // std::vector<int> devs(nDev);
  // for (int i = 0; i < nDev; ++i)
  // {
  //   devs[i] = i;
  // }
  int devs[2] = {0, 1};

  // allocating and initializing device buffers
  float **sendbuff = (float **)malloc(nDev * sizeof(float *));
  float **recvbuff = (float **)malloc(nDev * sizeof(float *));
  float **hptr = (float **)malloc(nDev * sizeof(float *));
  // create nDev streams for ndev devices
  cudaStream_t *s = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nDev);

  for (int i = 0; i < nDev; ++i)
  {
    CUDACHECK(cudaSetDevice(i));
    {
      // Device info
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, i);
      printf("\nDevice %d: \"%s\"%d,%d\n", i, deviceProp.name, deviceProp.major, deviceProp.minor);
    }
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s + i));
    init1<<<1, size>>>(sendbuff[i], i);
  }

  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

  // calling NCCL communication API. Group API is required when
  // using multiple devices per thread
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i)
  {
    NCCLCHECK(ncclAllReduce((const void *)sendbuff[i],
                            (void *)recvbuff[i], size, ncclFloat, ncclSum,
                            comms[i], s[i]));
  }
  NCCLCHECK(ncclGroupEnd());

  // synchronizing on CUDA streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i)
  {
    CUDACHECK(cudaSetDevice(i));
    // it will stall host until all operations are done
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }

  // free device buffers
  for (int i = 0; i < nDev; ++i)
  {
    CUDACHECK(cudaSetDevice(i));
    hptr[i] = (float *)malloc(size * sizeof(float));
    cudaMemcpy(hptr[i], recvbuff[i], size * sizeof(float), cudaMemcpyDeviceToHost);
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
  }

  // finalizing NCCL
  for (int i = 0; i < nDev; ++i)
  {
    ncclCommDestroy(comms[i]);
  }

  for (int i = 0; i < size; ++i)
  {
    for (int j = 0; j < nDev; ++j)
      std::cout<<"i= "<<i<<" "<<hptr[j][i]<<"\n";
    }
    free(hptr);
    printf("Success \n");
    return 0;
}