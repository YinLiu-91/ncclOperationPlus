//
// Example 1: Single Process, Single Thread, Multiple Devices
//

#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "ncclEnhance.h"
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

int main(int argc, char *argv[]) {
  std::cout << "================================================================"<<
               "\n    Executing " << argv[0] << " now!\n"<<
               "================================================================\n";
  ncclComm_t comms[2];

  // managing 2 devices
  int nDev = 2;
  const int size = 3;

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
    // 详见 https://gitee.com/liuyin-91/ncclexamples/blob/master/documents/nvdia%E5%AE%98%E6%96%B9documentation.md#%E4%BB%8E%E4%B8%80%E4%B8%AA%E7%BA%BF%E7%A8%8B%E7%AE%A1%E7%90%86%E5%A4%9A%E4%B8%AA-gpu 
    
    
    // NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; ++i)
    {
      CUDACHECK(cudaSetDevice(i));
      // work fine  
      // NCCLCHECK(ncclSend(sendbuff[i], size, ncclFloat, (i + 1) % 2, comms[i], s[i]));
      // NCCLCHECK(ncclRecv(recvbuff[i], size, ncclFloat, (i + 1) % 2, comms[i], s[i]));

      // work fine only if it is between ncclGroupStart() and ncclGroupEnd()
      // NCCLCHECK(ncclRecv(recvbuff[i], size, ncclFloat, (i + 1) % 2, comms[i], s[i]));
      // NCCLCHECK(ncclSend(sendbuff[i], size, ncclFloat, (i + 1) % 2, comms[i], s[i]));

      // 
      NCCLSendrecv(sendbuff[i],size,ncclFloat,(i + 1) % 2,recvbuff[i],size,comms[i],s[i]);

    }
    // NCCLCHECK(ncclGroupEnd());

    // synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        // it will stall host until all operations are done
        CUDACHECK(cudaStreamSynchronize(s[i]));
    }

    // free device buffers
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        hptr[i] = (float *)malloc(size * sizeof(float));
        cudaMemcpy(hptr[i], recvbuff[i], size * sizeof(float), cudaMemcpyDeviceToHost);
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
    }

    // finalizing NCCL
    for (int i = 0; i < nDev; ++i) {
        ncclCommDestroy(comms[i]);
    }

    for(int i=0;i<size;++i){
      for(int j=0;j<nDev;++j)
      std::cout<<"i= "<<i<<" "<<"hptr["<<i<<"] "<<hptr[j][i]<<"\n";
    }
    free(hptr);
    printf("Success \n");
    return 0;
}