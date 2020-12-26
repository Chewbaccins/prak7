#include <mpi.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
using namespace std;
#include "matrix_vect.h"
#include "step_alg.h"
#include "Ro_H.h"

int commRank, commSize;
Vector <int> basis;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);

    if (argc < 5) {
        if (commRank == 0) {
            cerr << "wrong number of parameters." << endl;
            cerr << "mpirun -np [number of processes] ./main [num qbits] [num steps] [min excited atoms] [max excited atoms]" << endl;
        }
        MPI_Finalize();
        exit(-1);
    }
    int numQBits = stoi(argv[1]),
        n = stoi(argv[2]),
        Emin = stoi(argv[3]),
        Emax = stoi(argv[4]);

    if ((Emin == Emax) || (Emin > Emax) || (Emin > numQBits) || (Emax > numQBits) || (Emin < 0) || (Emax < 0)) {
        if (commRank == 0) cerr << "Invalid Emin or Emax." << endl;
        MPI_Finalize();
        exit(-1);
    }

    int globalN = 0;
    for (int i = Emin; i <= Emax; i++)
        globalN += C(numQBits, i);

    if (commRank == 0) cout << "Basis" << endl;
    basis.resize(globalN);
    int j = 0;
    for (int state = 0; state < (int) pow(2, numQBits); state++) {
        if ((Energy(state) >= Emin) && (Energy(state) <= Emax)) {
            basis.data[j++] = state;
            if (commRank == 0) cout << binary_output(numQBits, state) << " ";
        }
    }
    if (commRank == 0) cout << endl;

    int context, Rows, Cols, proc_Row, proc_Col;
    int dims[2] = {0};
    MPI_Dims_create(commSize, 2, dims);
    Rows = dims[0];
    Cols = dims[1];
    if (Rows != Cols) {
        if (commRank == 0) cerr << "Square number of processes required" << endl;
        MPI_Finalize();
        exit(-1);
    }
    Cblacs_pinfo(&commRank, &commSize);
    Cblacs_get(-1, 0, &context);
    Cblacs_gridinit(&context, "Row-major", Rows, Cols);
    Cblacs_pcoord(context, commRank, &proc_Row, &proc_Col);

    int localN = ((globalN / Rows) * Rows == globalN) ? globalN / Rows : globalN /Rows + 1;
    int ISRCPROC_zero = 0;
    int proc_M = numroc_(&globalN, &localN, &proc_Row, &ISRCPROC_zero, &Rows);
    int proc_N = numroc_(&globalN, &localN, &proc_Col, &ISRCPROC_zero, &Cols);

    Vector <complexd> a(numQBits - 1);
    Vector <complexd> w(numQBits);
    Vector <complexd> phi(pow(2, numQBits));
    a.read("a.txt");
    w.read("w.txt");
    phi.read("phi.txt");
    phi.normal();
    
    Matrix_complexd ro(proc_M, proc_N, globalN, localN, context);
    initRo(ro, phi, proc_Row, proc_Col, numQBits, Emin, Emax, globalN, localN, proc_M, proc_N);
    if (commRank == 0) cout << "ro" << endl;
    ro.gather(Rows, Cols, proc_Row, proc_Col, globalN, localN);

    MPI_Barrier(MPI::COMM_WORLD);

    Matrix_complexd H(proc_M, proc_N, globalN, localN, context);
    initH(H, a, w, proc_Row, proc_Col, numQBits, Emin, Emax, globalN, localN, proc_M, proc_N);
    if (commRank == 0) cout << "H" << endl;
    H.gather(Rows, Cols, proc_Row, proc_Col, globalN, localN);

    MPI_Barrier(MPI::COMM_WORLD);

    if (commRank == 0) cout << "steps" << endl;
    step_alg(H, ro, n, context, proc_Row, proc_Col, Rows, Cols, globalN, localN, proc_M, proc_N);

    Cblacs_gridexit(context);
    MPI_Finalize();
}
