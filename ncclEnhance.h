#ifndef _NCCLENHANCE_H
#define _NCCLENHANCE_H
#include "nccl.h"
#include<cstddef>
// This function is copied from nccl 
static __inline__ int ncclTypeSize(ncclDataType_t type) {
  switch (type) {
    case ncclInt8:
    case ncclUint8:
      return 1;
    case ncclFloat16:
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
#endif
      return 2;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64:
      return 8;
    default:
      return -1;
  }
}


ncclResult_t NCCLSendrecv(void *sendbuff, size_t sendcount, ncclDataType_t datatype, int peer,
                          void *recvbuff,size_t recvcount,ncclComm_t comm, cudaStream_t stream)
{
    ncclGroupStart();
    auto a = ncclSend(sendbuff, sendcount, datatype, peer, comm, stream);
    auto b = ncclRecv(recvbuff, recvcount, datatype, peer, comm, stream);
    ncclGroupEnd();
    if (a||b)
    {
      if(a)
        return a;
      return b;
    }
    return ncclSuccess;
}

// Please be aware that sendcount,recvcount is the count for single rank
ncclResult_t NCCLAlltoall(void *sendbuff, size_t sendcount, ncclDataType_t senddatatype, void *recvbuff,
                         size_t recvcount, ncclDataType_t recvdatatype, ncclComm_t comm, cudaStream_t stream)
{
    ncclGroupStart();
    int nRanks;
    ncclCommCount(comm, &nRanks);
    for (int i = 0; i < nRanks; ++i)
    {
        auto a = NCCLSendrecv(static_cast<std::byte*>(sendbuff) + i * ncclTypeSize(senddatatype) * sendcount, sendcount, senddatatype, i, 
                    static_cast<std::byte*>(recvbuff) + i * ncclTypeSize(recvdatatype) * recvcount, recvcount, comm, stream);
        if (a)
            return a;
    }
    ncclGroupEnd();
    return ncclSuccess;
}

// Please be aware that sendcount,recvcount is the count for single rank
ncclResult_t NCCLGather(void *sendbuff, size_t sendcount, ncclDataType_t senddatatype, void *recvbuff,
                        size_t recvcount, ncclDataType_t recvdatatype, int root,ncclComm_t comm, 
                        cudaStream_t stream)
{
    ncclGroupStart();
    int myRank,nRanks;
    ncclCommUserRank(comm,&myRank);
    ncclCommCount(comm, &nRanks);
    auto a = ncclSend(sendbuff, sendcount, senddatatype, root, comm, stream);
    if(a){
        return a;
    }
    if(myRank==root){
        for(int i=0;i<nRanks;++i){
           auto b=ncclRecv( static_cast<std::byte*>(recvbuff)+i*ncclTypeSize(recvdatatype)*recvcount,recvcount,recvdatatype,i,comm,stream);
           if(b){
               return b;
           }
        }
    }
    ncclGroupEnd();
    return ncclSuccess;
}

// Please be aware that sendcount,recvcount is the count for single rank
ncclResult_t NCCLScatter(void *sendbuff, size_t sendcount, ncclDataType_t senddatatype, void *recvbuff,
                         size_t recvcount, ncclDataType_t recvdatatype, int root, 
                         ncclComm_t comm, cudaStream_t stream)
{
    ncclGroupStart();
    int myRank, nRanks;
    ncclCommUserRank(comm, &myRank);
    ncclCommCount(comm, &nRanks);
    if (myRank == root)
    {
        for (int i = 0; i < nRanks; ++i)
        {
            auto a = ncclSend(static_cast<std::byte*>(sendbuff) + i * ncclTypeSize(senddatatype) * sendcount, sendcount, recvdatatype, i, comm, stream);
            if (a)
                return a;
        }
    }
    auto b = ncclRecv(recvbuff, recvcount, recvdatatype, root, comm, stream);
    if (b)
        return b;
    ncclGroupEnd();
    return ncclSuccess;
}
#endif 