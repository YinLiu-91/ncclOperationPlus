# Introduction
When I reading [NVIDIA NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/mpi.html#other-collectives-and-point-to-point-operations), it said that `NCCL does not define specific verbs for sendrecv, gather, gatherv, scatter, scatterv, alltoall, alltoallv, alltoallw, nor neighbor collectives. All those operations can be simply expressed using a combination of ncclSend, ncclRecv, and ncclGroupStart/ncclGroupEnd, similarly to how they can be expressed with MPI_Isend, MPI_Irecv and MPI_Waitall.` So I try to use  [nccl's API](https://developer.nvidia.com/nccl) `ncclSend`,`ncclRecv`,`ncclGroupStart`,`ncclGroupEnd` to realize these function:

1. NCCLSendrecv
1. NCCLGather
1. NCCLScatter
1. NCCLAlltoall

I referenced [openmpi's API](https://mpitutorial.com/tutorials/) when writing these APIs.

# Build
1. Use a linux PC
1. Make sure that openmpi and nccl is installed on your PC
1. Make sure your cuda version or nvcc could use std::c++17 (my cuda version is 11.4 )
1. Clone this repo to your disk
1. `cd` to any one of three directories and then:
    - `make` or `make all` , it will build the binary file
    - `make test`, it will execute the examples
 