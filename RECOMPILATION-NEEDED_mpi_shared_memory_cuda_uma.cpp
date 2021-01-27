/*
----------------------------------------------------------------------
!!! CAUTION !!!
This code requires you to re-compile open-mpi.
See the README.md for more informations
----------------------------------------------------------------------
*/

/*
compile using :
mpic++ mpi_rma_cuda_uma.cpp -fopenmp -foffload=nvptx-none -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -fno-stack-protector -fcf-protection=none

and run using :
RECOMPILED_OMPI_PATH/mpirun a.out
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

int MPI_Win_create_shared(void *base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, MPI_Win *win) {
  int err = MPI_Win_create(base, size, disp_unit, info, comm, win);
  if (err != 0)
    return err;

  int flavor = MPI_WIN_FLAVOR_SHARED;
  return MPI_Win_set_attr(*win, MPI_WIN_CREATE_FLAVOR, &flavor);
}

void print(int t[],int s) {
  for (int i = 0; i < s; i++) {
    printf("%d ",t[i]);
  }
    printf("\n");
}

int main(int argc, char **argv) {
  int size, rank, nodesize, noderank;
  int *table;

  MPI_Comm nodecomm;

  MPI_Aint winsize;
  int windisp;
  int *winprintr;
  MPI_Win wintable;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* Create node-local communicator */
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodecomm);

  MPI_Comm_size(nodecomm, &nodesize);
  MPI_Comm_rank(nodecomm, &noderank);

  int gpuid = 0;
  const int table_size = 500000000;
  const int print_size = 10;

  /*
  --------------------------------------------------
  important part
  --------------------------------------------------
  */
  if (noderank == 0) {
    /* We first allocate a uma buffer using cudaMallocManaged */
    cudaMallocManaged((CUdeviceptr**)&table, table_size*sizeof(int), CU_MEM_ATTACH_GLOBAL);
    /* we then create a shared window using this buffer */
    MPI_Win_create_shared(table, table_size*sizeof(int), sizeof(int), MPI_INFO_NULL, nodecomm, &wintable); // analogous to MPI_Win_create
  } else {
    /*
    Only rank 0 on a node actually allocates memory
    We then get the actual shared table pointer using MPI_Win_shared_query
    */
    MPI_Win_create_shared(table, 0, sizeof(int), MPI_INFO_NULL, nodecomm, &wintable);
    MPI_Win_shared_query(wintable, 0, &winsize, &windisp, &table);
  }
  /* we finish by making table an omp device pointer using omp_target_associate_ptr */
  omp_target_associate_ptr(table, table, table_size*sizeof(int), 0 /* device offset */ , gpuid);
  /*
  --------------------------------------------------
  important part
  --------------------------------------------------
  */

  printf("Rank %d of %d, in-node: rank %d of %d\n", rank, size, noderank, nodesize);

  // Initialise table on rank 0 with appropriate synchronisation
  if (noderank == 0) {
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < table_size; i++)
      table[i] = i;
  }
  MPI_Win_fence(0, wintable);

  // print it
  printf("(first) rank %d, noderank %d, ", rank, noderank);  print(table, print_size);

  // modify the table from each node on cpu
  MPI_Win_fence(0, wintable);
  if (noderank == 1) {
    #pragma omp parallel for
    for (int i = 0; i < table_size; i++)
      table[i] -= 2*i;
  }
  MPI_Win_fence(0, wintable);

  // print it
  printf("(second) rank %d, noderank %d, ", rank, noderank); print(table, print_size);

  /*
  --------------------------------------------------
  important part
  --------------------------------------------------
  */
  /* free the window */
  MPI_Win_free(&wintable);
  if (noderank == 0) {
    /* free the cuda buffer */
    cudaFree(table);
  }
  /*
  --------------------------------------------------
  important part
  --------------------------------------------------
  */

  MPI_Finalize();
}
