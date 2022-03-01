//
// Example 2: One Device Per Process Or Thread
//


#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <printf.h>
#ifdef __linux
#include <unistd.h>
#endif

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

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

static uint64_t getHostHash(const char *string) {
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) + string[c];
    }
    return result;
}

static void getHostName(char *hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i = 0; i < maxlen; i++) {
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

int main(int argc, char *argv[]) {
    {
        char host[256];
#ifdef __linux
        printf("PID %d on node %s is ready for attach\n",
               getpid(), host);
        fflush(stdout);
#endif
        if(argc!=2){
            std::cout<<"Please input a int,'0' means for debug,'1' means execute directly\n";
        }
        if (std::stoi(argv[1]) == 0)
        {
            {
                int i = 0;
                while (i == 0)
                {
                    i = 0;
                }
            }
        }
    }

    int size = 2;
    int myRank, nRanks, localRank = 0;
    // initializing MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));
    if (myRank == 0)
    {
        std::cout << "================================================================"
                  << "\n    Executing " << argv[0] << " now!\n"
                  << "================================================================\n";
    }
    std::cout<<"Rank: "<<myRank<<"\n";
    {
        int deviceCount = 0;
        cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
        for (int dev = 0; dev < deviceCount; ++dev)
        {
            cudaSetDevice(dev);
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, dev);
            printf("\nDevice %d: \"%s\"%d,%d\n", dev, deviceProp.name,deviceProp.major,deviceProp.minor);
        }
    }

    // calculating localRank based on hostname which is used in
    // selecting a GPU
    uint64_t hostHashs[nRanks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                           hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int p = 0; p < nRanks; p++) {
        if (p == myRank) {
            break;
        }
        if (hostHashs[p] == hostHashs[myRank]) {
            localRank++;
        }
    }

    ncclUniqueId id;
    ncclComm_t comm;
    float *sendbuff;
    cudaStream_t s;

    if (myRank == 0) {
        ncclGetUniqueId(&id);
    }
    MPICHECK(MPI_Bcast((void *) &id, sizeof(id), MPI_BYTE, 0,
                       MPI_COMM_WORLD));

    // picking a GPU based on localRank, allocate device buffers
    CUDACHECK(cudaSetDevice(localRank));
    CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(&s));

    // call init kernel to init rank 0 sendbuff data
    if (myRank == 0)
        init<<<1, size>>>(sendbuff, myRank);

    // malloc host mem
    float *hptr = (float *)malloc(size * sizeof(float));
    cudaMemcpy(hptr,sendbuff,size*sizeof(float),cudaMemcpyDeviceToHost);
    std::cout << "sendbuff-before-brdcast:\n";
    for (int i = 0; i < size; ++i)
    {
        std::cout << "myRank: " << myRank << " hptr["<<i<<"]: " << hptr[i] << "\n";
    }
    // initializing NCCL
    NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

    // communicating using NCCL
    NCCLCHECK(ncclBcast((void *)sendbuff,
                         size, ncclFloat, 0,
                         comm, s));
                         
    cudaMemcpy(hptr,sendbuff,size*sizeof(float),cudaMemcpyDeviceToHost);
    std::cout << "sendbuff-after-brdcast:\n";
    for (int i = 0; i < size; ++i)
    {
        std::cout << "myRank: " << myRank << " hptr["<<i<<"]: " << hptr[i] << "\n";
    }

    // completing NCCL operation by synchronizing on the CUDA stream
    CUDACHECK(cudaStreamSynchronize(s));
    // free device buffers
    CUDACHECK(cudaFree(sendbuff));

    // finalizing NCCL
    ncclCommDestroy(comm);

    // finalizing MPI
    MPICHECK(MPI_Finalize());
    free(hptr);
    printf("[MPI Rank %d] Success \n", myRank);
    // cudaDeviceSynchronize();
    return 0;
}