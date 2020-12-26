#include <mpi.h>
#include <cstdlib>
#include <iostream>
#include <complex>
#include "step_alg.h"

using namespace std;

void step_alg(Matrix_complexd &H, Matrix_complexd &ro, int globalN, int localN, int proc_M, int proc_N,
                                                                int context, int proc_Row, int proc_Col, 
                                                                int commRank, int commSize, 
                                                                int Rows, int Cols,  int n) {
    int one = 1, info = 0;
    Matrix_complexd tmp(proc_M, proc_N, globalN, localN, context);
    Matrix_complexd Z(proc_M, proc_N, globalN, localN, context);
    Vector_complexd w(globalN);

    char jobz = 'V', uplo = 'U';
    int workspace = globalN * globalN; 
    if (workspace < globalN * 100) {
        workspace = globalN * 100;
    };
    Vector_complexd work(workspace);
    Vector_complexd rwork(workspace);
    Vector_complexd iwork(workspace);

    pzheevd_(&jobz, &uplo, &globalN,
        H.data, &one, &one, H.desc,
        w.data,
        Z.data, &one, &one, Z.desc,
        work.data, &workspace, rwork.data, &workspace, iwork.data, &workspace,
        &info); // compute eigenvalues and eigenvectors
    if (info != 0) {
        cerr << "Error on pzheevd_" << endl;
        MPI_Finalize();
        exit(-1);
    }

    Matrix_complexd expW(proc_M, proc_N, globalN, localN, context);
    Vector_int iPiv(globalN);
    Matrix_complexd U(proc_M, proc_N, globalN, localN, context);
    Matrix_complexd roEvol(proc_M, proc_N, globalN, localN, context);
    double dT = 0.01;
    // start main part
    for (int t = 1; t <= n; t++) {
        complexd mul(0, t * dT * (-1));
        for (int i = 0; i < proc_M; i++) {
            for (int j = 0; j < proc_N; j++) {
                if (localN * proc_Row + i == localN * proc_Col + j) {
                    expW.data[i * proc_N + j] = exp(w.data[localN * proc_Row + i] * mul);
                } else {
                    complexd(0, 0);
                }
            }
        }
        complexd alpha(1.0, 0.0), beta(0.0, 0.0);
        char no = 'N', conj = 'C';
        pzgemm_(&no, &no, &globalN, &globalN, &globalN,
            &alpha, Z.data, &one, &one, Z.desc,
                    expW.data, &one, &one, expW.desc,
            &beta,  tmp.data, &one, &one, tmp.desc
        );
        pzgemm_(&no, &conj, &globalN, &globalN, &globalN,
            &alpha, tmp.data, &one, &one, tmp.desc,
                    Z.data, &one, &one, Z.desc,
            &beta,  U.data, &one, &one, U.desc
        );
        pzgemm_(&no, &no, &globalN, &globalN, &globalN,
            &alpha, U.data, &one, &one, U.desc,
                    ro.data, &one, &one, ro.desc,
            &beta,  tmp.data, &one, &one, tmp.desc
        );
        pzgemm_(&no, &conj, &globalN, &globalN, &globalN,
            &alpha, tmp.data, &one, &one, tmp.desc,
                    U.data, &one, &one, U.desc,
            &beta, roEvol.data, &one, &one, roEvol.desc
        );

        if (proc_Row == proc_Col) {
            //cout << "trying output" << endl;
            if (commRank != commSize - 1) {
                Vector_complexd localTrace(localN);
                for (int i = 0; i < localN; i++) {
                    localTrace.data[i] = roEvol.data[i * localN + i];
                }
                MPI_Send(localTrace.data, localN, MPI_DOUBLE_COMPLEX, commSize - 1, proc_Row, MPI_COMM_WORLD);
            } else {
                Vector_complexd roTrace(globalN);
                for (int i = 0; i < Rows - 1; i++) {
                    MPI_Recv(roTrace.data + i * localN, localN, MPI_DOUBLE_COMPLEX, i * Cols + i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                for (int i = 0; i < proc_M; i++) roTrace.data[localN * (Rows - 1) + i] = roEvol.data[i * proc_N + i];
                cout << "tr(ro(" << t << " * dT)): ";
                for (int i = 0; i < globalN; i++) cout << abs(roTrace.data[i]) << " ";
                cout << endl;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}