//
// Multiple Devices per Thread
// 
// 

#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <iostream>
#include "ncclEnhance.h"

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

__global__ void  init(float *dptr,int myRank)
{
  int id = threadIdx.x;
  dptr[id] = id;
//   printf("kernel-myRank: %d id: %f\n",myRank,dptr[id]);
}


int main(int argc, char* argv[])
{
    //each process is using two GPUs
    int nDev = 1;
    int size = 6;

    int myRank, nRanks, localRank = 0;

    //initializing MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    if (myRank == 0)
    {
        std::cout << "================================================================"
                  << "\n    Executing " << argv[0] << " now!\n"
                  << "================================================================\n";
    }

    //calculating localRank which is used in selecting a GPU
    uint64_t hostHashs[nRanks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int p = 0; p < nRanks; p++)
    {
      if (p == myRank)
        break;
      if (hostHashs[p] == hostHashs[myRank])
        localRank++;
  }
    std::cout<<"myRank: "<<myRank<<" localRank: "<<localRank<<"\n";

    float **sendbuff = (float **)malloc(nDev * sizeof(float *));
    float **recvbuff = (float **)malloc(nDev * sizeof(float *));
    float **hptr = (float **)malloc(nDev * sizeof(float *));
    cudaStream_t *s = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nDev);

    //picking GPUs based on localRank
    for (int i = 0; i < nDev; ++i)
    {
      CUDACHECK(cudaSetDevice(localRank * nDev + i)); // 给所有设备编号
      CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
      CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
      CUDACHECK(cudaMemset(sendbuff[i], 0, size * sizeof(float)));
      CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
      CUDACHECK(cudaStreamCreate(s + i));
      hptr[i] = (float *)malloc(size * sizeof(float));
  }


  ncclUniqueId id;
  ncclComm_t comms[nDev];


  //generating NCCL unique ID at one process and broadcasting it to all
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  //initializing NCCL, group API is required around ncclCommInitRank as it is
  //called across multiple GPUs in each thread/process
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; i++)
  {
    CUDACHECK(cudaSetDevice(localRank * nDev + i));
    init<<<1,  size >>>(sendbuff[i], myRank);
    NCCLCHECK(ncclCommInitRank(comms + i, nRanks * nDev, id, myRank * nDev + i));
    cudaMemcpy(hptr[i], sendbuff[i], size * sizeof(float), cudaMemcpyDeviceToHost);
      std::cout << "myRank" << myRank << " sendbuff[" << i << "]"
                << "\n";
      for (int j = 0; j < size; ++j)
      {
        std::cout << " j: " << j << " hptr[i][j]: " << hptr[i][j] << "\n";
      }
  }
  NCCLCHECK(ncclGroupEnd());


  // scatter Data
  int sendsize=size/(nRanks*nDev);
  for(int i=0;i<nDev;++i){
    NCCLAlltoall(sendbuff[i],sendsize,ncclFloat,recvbuff[i],sendsize,ncclFloat,comms[i],s[i]);
  }

  for (int i = 0; i < nDev; ++i)
  {
    cudaMemcpy(hptr[i], recvbuff[i],size * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout<<"myRank"<<myRank<<" recvbuff["<<i<<"]"<<"\n";
    for(int j=0;j<size;++j){
        std::cout<<" j: "<<j<<" hptr[i][j]: "<<hptr[i][j]<<"\n";
    }
  }

  //synchronizing on CUDA stream to complete NCCL communication
  for (int i=0; i<nDev; i++)
      CUDACHECK(cudaStreamSynchronize(s[i]));


  //freeing device memory
  for (int i=0; i<nDev; i++) {
     CUDACHECK(cudaFree(sendbuff[i]));
     CUDACHECK(cudaFree(recvbuff[i]));
     free(hptr[i]);
  }


  //finalizing NCCL
  for (int i=0; i<nDev; i++) {
     ncclCommDestroy(comms[i]);
  }


  //finalizing MPI
  MPICHECK(MPI_Finalize());


  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}