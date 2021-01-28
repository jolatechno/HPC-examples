/*
compile using :
g++ omp_uma.cpp -fopenmp -foffload=nvptx-none -fno-stack-protector -fcf-protection=none
*/

#include <omp.h>
#include <stdio.h>
#include <unistd.h>
#include <cstdlib>

void print(int t[],int s) {
  for (int i = 0; i < s; i++) {
    printf("%d ",t[i]);
  }
    printf("\n");
}

int main(int argc, char **argv) {
  const int size = 500000000;
  const int print_size = 10;

  /*
  --------------------------------------------------
  important part
  --------------------------------------------------
  */
  /* allocate a uma buffer using malloc (or other) */
  #pragma omp extension unified_memory
  int *table = (int*)malloc(size*sizeof(int));
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
  free(table);
  /*
  --------------------------------------------------
  important part
  --------------------------------------------------
  */

  return 0;
}
