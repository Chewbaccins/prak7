#include <mpi.h>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <complex>
#include "step_alg.h"

using namespace std;

int main(int argc, char **argv)
{
    int commRank, commSize;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    
    if (argc < 4) {
        if (commRank == 0) cerr << "wrong number of arguments" << endl;
        MPI_Finalize();
        exit(-1);
    }
    
    int context, Rows, Cols, proc_Row, proc_Col;
    int dims[2] = {0};    
    MPI_Dims_create(commSize, 2, dims);
    Rows = dims[0];
    Cols = dims[1];
    Cblacs_pinfo(&commRank, &commSize);
    Cblacs_get(-1, 0, &context);
    Cblacs_gridinit(&context, "Row-major", Rows, Cols);
    Cblacs_pcoord(context, commRank, &proc_Row, &proc_Col);
    int globalN = stoi(argv[1]),
        localN = stoi(argv[2]),
        n = stoi(argv[3]);
    int ISRCPROC_zero = 0;
    int proc_M = numroc_(&globalN, &localN, &proc_Row, &ISRCPROC_zero, &Rows);
    int proc_N = numroc_(&globalN, &localN, &proc_Col, &ISRCPROC_zero, &Cols);

    Matrix_complexd H(proc_M, proc_N, globalN, localN, context);
    Matrix_complexd ro(proc_M, proc_N, globalN, localN, context);
    H.readMatrix(globalN, localN, proc_M, proc_N, proc_Row, proc_Col, "H");
    ro.readMatrix(globalN, localN, proc_M, proc_N, proc_Row, proc_Col, "ro");

    step_alg(H, ro, globalN, localN, proc_M, proc_N, context, proc_Row, proc_Col, commRank, commSize, Rows, Cols, n);

    Cblacs_gridexit(context);
    MPI_Finalize();
}
