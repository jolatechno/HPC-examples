# HPC examples

This is a repository where I put simple example showcasing the basic capabilities of __MPI__, __OPENMP__ and __CUDA__.

I choose to make a repository out of them because some of those capability aren't particularly well documented, and those simple examples might save you the time I took making them work properly.

## UMA with CUDA and OPENMP

I demonstrate in [uma_cuda_omp.cpp](./uma_cuda_omp.cpp) how you can allocate a buffer using Unified Memory Access (or __UMA__) using __CUDA__, and then access it from both the cpu and an __OPENMP__ `target` section.

The important section is the following :

```cpp
/*
We first allocate a uma buffer using cudaMallocManaged
And then make it a omp device pointer using omp_target_associate_ptr
*/
cudaMallocManaged((CUdeviceptr**)&table, size*sizeof(int), CU_MEM_ATTACH_GLOBAL);
omp_target_associate_ptr(table, table, size*sizeof(int), 0 /* device offset */ , gpuid);
```

To then free `ptr` you can use :

```cpp
/* Free memory */
cudaFree(ptr);
```

To compile this example, you can use `nvcc` or using `g++` with :

```shell
g++ uma_cuda_omp.cpp -fopenmp -foffload=nvptx-none -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -fno-stack-protector -fcf-protection=none
```

### Equivalent code using only OPNEMP

__UMA__ is also possible using only __OPNEMP__ as shown in [omp_uma.cpp](./omp_uma.cpp) using the following code :

```cpp
/* allocate a uma buffer using malloc (or other) */
#pragma omp extension unified_memory
int *table = (int*)malloc(size*sizeof(int));
```

`table` can then be freed as usual:

```cpp
/* Free memory */
free(table);
```

## Shared memory using MPI

[mpi_shared_window.cpp](./mpi_shared_window.cpp) shows an example of a shared window in __MPI__.

The important section is the following :

```cpp
/*
We first allocate a shared window using MPI_Win_allocate_shared
Only rank 0 on a node actually allocates memory
We then get the actual shared table pointer using MPI_Win_shared_query
*/
MPI_Win_allocate_shared(noderank == 0 ? nodesize*sizeof(int) : 0,
  sizeof(int), MPI_INFO_NULL, nodecomm, &table, &wintable);
if (noderank != 0) MPI_Win_shared_query(wintable, 0, &winsize, &windisp, &table);
```

To then free the shared window you can use :

```cpp
/* free the memory */
MPI_Win_free(&wintable);
```

You can compile this example using `mpic++` and run it using `mpirun`.

## RMA with MPI and UMA with CUDA and OPENMP

I demonstrate in [mpi_rma_cuda_uma.cpp](./mpi_rma_cuda_uma.cpp) how you can allocate a buffer using Unified Memory Access (or __UMA__) using __CUDA__, and then access it from an other node using Remote Memory Access (or __RMA__) using __MPI__.

The important section is the following :

```cpp
if (rank == 0) {
  /*
  We first allocate a uma buffer using cudaMallocManaged on the first rank
  And then make it a omp device pointer using omp_target_associate_ptr
  */
  cudaMallocManaged((CUdeviceptr**)&table, table_size*sizeof(int), CU_MEM_ATTACH_GLOBAL);
  omp_target_associate_ptr(table, table, table_size*sizeof(int), 0 /* device offset */ , gpuid);
} else {
  /* we do a normal alloc on the second node */
  table = (int*)malloc(table_size*sizeof(int));
}
/* We then allocate a window using MPI_Win_create */
MPI_Win_create(table, table_size*sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &wintable);
```

You can then use `MPI_Get` and `MPI_Put` (demonstrated in [mpi_rma_cuda_uma.cpp](./mpi_rma_cuda_uma.cpp)) to remotely access `table`.

To then free both the window and both buffer, you can use:

```cpp
/* free the window */
MPI_Win_free(&wintable);
if (rank == 0) {
  /* free the cuda buffer */
  cudaFree(table);
} else {
  /* free the normal buffer */
  free(table);
}
```

To compile this example, you can use `mpic++` with :

```shell
mpic++ mpi_rma_cuda_uma.cpp -fopenmp -foffload=nvptx-none -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -fno-stack-protector -fcf-protection=none
```

And you can run it using :

```shell
mpirun -np 2 a.out
```
