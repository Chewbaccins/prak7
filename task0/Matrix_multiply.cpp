#include <mpi.h>
#include <iomanip>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <complex>
// #include "scalapack.h"

using namespace std;
typedef complex<double> complexd;

extern "C" {
    // Cblacs declarations
    void Cblacs_pinfo(int*, int*);
    void Cblacs_get(int, int, int*);
    void Cblacs_gridinit(int*, const char*, int, int);
    void Cblacs_pcoord(int, int, int*, int*);
    void Cblacs_gridexit(int);
    void Cblacs_exit(int);
    void Cblacs_barrier(int, const char*);
    void Cdgerv2d(int, int, int, double*, int, int, int);
    void Cdgesd2d(int, int, int, double*, int, int, int);
    int numroc_(int*, int*, int*, int*, int*);
    void descinit_(int *idescal, int *m, int *n, int *mb, int *nb, int *dummy1 , int *dummy2 , int *icon, int *procRows, int *info);
    void pdgemm_(char *transa, char *transb, int *M, int *N, int *K, double *alpha, double *A, int *ia, int *ja, int *descA,
            double *B, int *ib, int *jb, int *descB, double *beta, double *C, int *ic, int *jc, int *descC);
    void pzgemm_(char *transa, char *transb, int *M, int *N, int *K, complexd *alpha, complexd *A, int *ia, int *ja, int *descA,
            complexd *B, int *ib, int *jb, int *descB, complexd *beta, complexd *C, int *ic, int *jc, int *descC);
}

class Matrix_compl {
public:
  
  complexd *data;
  
  Matrix_compl(int N = 10, int M = 10) {
    data = new complexd[N * M];
    if (data == NULL) {
      cerr << "Error on matrix allocation" << endl;
      MPI_Finalize();
      exit(-1);
    }
  }
  
  void genMatrix(int M, int N) {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        data[i * N + j] = complexd(rand() / (double) RAND_MAX, rand() / (double) RAND_MAX);
      }
    }
  }
  
  ~Matrix_compl() {
    delete[] data;
  }
};

class Matrix_double {
public:

  double *data;

  Matrix_double(int N = 10, int M = 10) {
    data = new double[N * M];
    if (data == NULL) {
      cerr << "Error on matrix allocation" << endl;
      MPI_Finalize();
      exit(-1);
    }
  }

  void genMatrix(int M, int N) {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        data[i * N + j] = (double) rand() / RAND_MAX;
      }
    }
  }

  ~Matrix_double() {
    delete[] data;
  }
};

int main(int argc, char **argv)
{
  int commRank, commSize, dims[2] = {0},
      gridRows, gridCols, proc_Row, proc_Col;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
  MPI_Comm_size(MPI_COMM_WORLD, &commSize);

  int *seeds = NULL;
	if (commRank == 0) {
		srand(time(NULL));
		seeds = new int[commSize];
		for (int i = 0; i < commSize; i++) {
			seeds[i] = rand();
		}
	}
	int recv_seed;
	MPI_Scatter(seeds, 1, MPI_INTEGER, &recv_seed, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
	if (commRank == 0) {
		delete[] seeds;
	}
	srand(recv_seed);

  if (argc != 4) {
    if (commRank == 0) cerr << "wrong number of command line parameters\n";
    MPI_Finalize();
    exit(-1);
  }
  int globalN = std::stoi(argv[1]),
      localN = std::stoi(argv[2]);
  char charType = argv[3][0];

  MPI_Dims_create(commSize, 2, dims);
  gridRows = dims[0];
  gridCols = dims[1];
  int context;
  Cblacs_pinfo(&commRank, &commSize);
  Cblacs_get(-1, 0, &context);
  Cblacs_gridinit(&context, "Row-major", gridRows, gridCols);
  Cblacs_pcoord(context, commRank, &proc_Row, &proc_Col);
  int ISRCPROC_zero = 0;
  int proc_M = numroc_(&globalN, &localN, &proc_Row, &ISRCPROC_zero, &gridRows); 
  int proc_N = numroc_(&globalN, &localN, &proc_Col, &ISRCPROC_zero, &gridCols); 
  int descA[9], descB[9], descC[9], info[3] = {0};
  descinit_(descA, &globalN, &globalN, &localN, &localN, &ISRCPROC_zero, &ISRCPROC_zero, &context, &proc_M, &info[0]);
  descinit_(descB, &globalN, &globalN, &localN, &localN, &ISRCPROC_zero, &ISRCPROC_zero, &context, &proc_M, &info[1]);
  descinit_(descC, &globalN, &globalN, &localN, &localN, &ISRCPROC_zero, &ISRCPROC_zero, &context, &proc_M, &info[2]);
  if (info[0] * info[1] * info[2] != 0) {
    cerr << "Error on descinit_" << endl;
    Cblacs_gridexit(context);
    MPI_Finalize();
    exit(-1);
  }
  //cout << "char_type" << charType << endl;
  if (charType == 'z') {
    Matrix_compl localA(proc_M, proc_N);
    Matrix_compl localB(proc_M, proc_N);
    Matrix_compl localC(proc_M, proc_N);
    localA.genMatrix(proc_M, proc_N);
    localB.genMatrix(proc_M, proc_N);
    complexd alpha(1.0, 0.0), beta(0.0, 0.0);
    char no = 'N';
    int one = 1;
    double start = MPI_Wtime();
    pzgemm_(&no, &no, &globalN, &globalN, &globalN,
        &alpha,
        localA.data, &one, &one, descA,
        localB.data, &one, &one, descB,
        &beta,
        localC.data, &one, &one, descC);
    double time_result = MPI_Wtime() - start;
    if (commRank == 0)
      cout << "time result " << time_result << endl;
  }
  else {
    //cout << proc_M << " " << proc_N << endl;
    Matrix_double localA(proc_M, proc_N);
    Matrix_double localB(proc_M, proc_N);
    Matrix_double localC(proc_M, proc_N);
    localA.genMatrix(proc_M, proc_N);
    localB.genMatrix(proc_M, proc_N);
    double alpha = 1.0, beta = 0.0;
    char no = 'N';
    int one = 1;
    double start = MPI_Wtime();
    pdgemm_(&no, &no, &globalN, &globalN, &globalN,
        &alpha,
        localA.data, &one, &one, descA,
        localB.data, &one, &one, descB,
        &beta,
        localC.data, &one, &one, descC);
    double time_result = MPI_Wtime() - start;
    if (commRank == 0)
      cout << "time result " << time_result << endl;
  }

  if (info[0] != 0) {
    cerr << "Error on multiplication" << endl;
    Cblacs_gridexit(context);
    MPI_Finalize();
    exit(-1);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  Cblacs_gridexit(context);
  MPI_Finalize();
  return 0;
}
