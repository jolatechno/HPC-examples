/*
compile using :
g++ uma_cuda_omp.cpp -fopenmp -foffload=nvptx-none -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -fno-stack-protector -fcf-protection=none

or nvcc
*/

#include <omp.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>

void print(int t[],int s) {
  for (int i = 0; i < s; i++) {
    printf("%d ",t[i]);
  }
    printf("\n");
}

int main(int argc, char **argv) {
  const int size = 500000000;
  const int print_size = 10;
  int *table;
  int gpuid = 0;

  /*
  --------------------------------------------------
  important part
  --------------------------------------------------
  */
  /*
  We first allocate a uma buffer using cudaMallocManaged
  And then make it a omp device pointer using omp_target_associate_ptr
  */
  cudaMallocManaged((CUdeviceptr**)&table, size*sizeof(int), CU_MEM_ATTACH_GLOBAL);
  omp_target_associate_ptr(table, table, size*sizeof(int), 0 /* device offset */ , gpuid);
  /*
  --------------------------------------------------
  important part
  --------------------------------------------------
  */

  /* cpu loop */
  #pragma omp parallel for
  for (int i = 0; i < size; i++)
    table[i] = i;

  /* print */
  print(table, print_size);

  /* prefetch, and sleep to see the impact on gpu mem */
  cudaMemPrefetchAsync(table, size * sizeof(int), 0);
  sleep(2);

  /* cpu loop */
  #pragma omp target teams distribute parallel for
  for (int i = 0; i < size; i++)
    table[i] -= 2*i;

  /* print */
  print(table, print_size);

  /*
  --------------------------------------------------
  important part
  --------------------------------------------------
  */
  /* Free memory */
  cudaFree(table);
  /*
  --------------------------------------------------
  important part
  --------------------------------------------------
  */

  return 0;
}
