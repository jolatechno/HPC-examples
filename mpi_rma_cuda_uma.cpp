/*
compile using :
mpic++ mpi_rma_cuda_uma.cpp -fopenmp -foffload=nvptx-none -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -fno-stack-protector -fcf-protection=none

and run using :
mpirun a.out
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

void print(int t[],int s) {
  for (int i = 0; i < s; i++) {
    printf("%d ",t[i]);
  }
    printf("\n");
}

int main(int argc, char **argv) {
  int size, rank;
  int *table;

  MPI_Aint winsize;
  int *winprintr;
  MPI_Win wintable;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int gpuid = 0;
  const int table_size = 500000000;
  const int print_size = 10;

  /*
  --------------------------------------------------
  important part
  --------------------------------------------------
  */
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
  /*
  --------------------------------------------------
  important part
  --------------------------------------------------
  */

  printf("Rank %d of %d\n", rank, size);

  // write using gpu
  if (rank == 0) {
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < table_size; i++)
      table[i] = i;
  }
  MPI_Win_fence(0, wintable);

  if (rank == 1) {
    // get
    MPI_Get(&table[0], table_size, MPI_INT, 0, 0 /* offset */, table_size, MPI_INT, wintable);

    //print
    printf("(MPI_Get from gpu) "); print(table, print_size);

    // assign
    #pragma omp parallel for
    for (int i = 0; i < table_size; i++)
      table[i] -= 2*i;

    // put
    MPI_Put(&table[0], table_size, MPI_INT, 0, 0 /* offset */, table_size, MPI_INT, wintable);
  }

  MPI_Win_fence(0, wintable);

  if (rank == 0) {
    printf("(MPI_Put to gpu) "); print(table, print_size);
  }

  /*
  --------------------------------------------------
  important part
  --------------------------------------------------
  */
  /* free the window */
  MPI_Win_free(&wintable);
  if (rank == 0) {
    /* free the cuda buffer */
    cudaFree(table);
  } else {
    /* free the normal buffer */
    free(table);
  }
  /*
  --------------------------------------------------
  important part
  --------------------------------------------------
  */

  MPI_Finalize();
}
