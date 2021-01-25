/*
compile using :
mpic++ mpi_shared_window

and run using:
mpirun a.out
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

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

  /*
  --------------------------------------------------
  important part
  --------------------------------------------------
  */
  /*
  We first allocate a shared window using MPI_Win_allocate_shared
  Only rank 0 on a node actually allocates memory
  We then get the actual shared table pointer using MPI_Win_shared_query
  */
  MPI_Win_allocate_shared(noderank == 0 ? nodesize*sizeof(int) : 0,
    sizeof(int), MPI_INFO_NULL, nodecomm, &table, &wintable);
  if (noderank != 0) MPI_Win_shared_query(wintable, 0, &winsize, &windisp, &table);
  /*
  --------------------------------------------------
  important part
  --------------------------------------------------
  */

  printf("Rank %d of %d, in-node: rank %d of %d\n", rank, size, noderank, nodesize);

  // Initialise table on rank 0 with appropriate synchronisation
  if (noderank == 0)
    for (int i = 0; i < nodesize; i++)
      table[i] = -1;
  MPI_Win_fence(0, wintable);

  // print it
  printf("(first) rank %d, noderank %d, ", rank, noderank);  print(table, nodesize);

  // modify the table from each node
  MPI_Win_fence(0, wintable);
  table[noderank] = noderank;
  MPI_Win_fence(0, wintable);

  // print it
  printf("(second) rank %d, noderank %d, ", rank, noderank); print(table, nodesize);

  /*
  --------------------------------------------------
  important part
  --------------------------------------------------
  */
  /* free the memory */
  MPI_Win_free(&wintable);
  /*
  --------------------------------------------------
  important part
  --------------------------------------------------
  */

  MPI_Finalize();
}
