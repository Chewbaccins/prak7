#include <mpi.h>
#include <cstdlib>
#include <iostream>
#include <complex>
#include <cmath>
using namespace std;
#include "declares.h"
#include "matrix_vector.h"

int C(int n, int k) { // C from n per k
    int B[n + 1][n + 1];
    for (int i = 0; i <= n; ++i) {
       B[i][0] = 1;
       B[i][i] = 1;
       for (int j = 1; j < i; ++j) {
           B[i][j]=B[i - 1][j - 1] + B[i - 1][j];
       }
    }
    return B[n][k];
}

string binary_output(int numQBits, int classicState) {
    string binary;
    for (int i = 0; i < numQBits; ++i) {
        binary = ((classicState % 2 == 0) ? "0" : "1") + binary;
        classicState >>= 1;
    }
    return binary;
}

complexd sumActiveQBits(complexd *wData, int numQBits, int classicState) {
    int mask = (int) pow(2, numQBits);
    complexd sum;

    for (int numQBit = 0; numQBit < numQBits; ++numQBit) {
        mask >>= 1;
        if ((classicState & mask) != 0) {
            sum += wData[numQBit];
        }
    }
    // if (commRank == 0) cout << sum << endl; DBG
    return sum;
}

bool areNeighbours(int classicState1, int classicState2) {
    int mask = classicState1 ^ classicState2;
    if (mask % 3 != 0) return false;
    int div = mask / 3;
    if (mask / div != 3) return false;
    double tmp = 0.0;
    if (modf(log2((double) div), &tmp) != 0.0d) return false;
    return true;
}

int Energy(int classicState) { // number of ones in binary classicState
    int E = 0;
    while (classicState != 0) {
        E += classicState & 1;
        classicState >>= 1;
    }
    return E;
}

void initH(Matrix_complexd &H, Vector <complexd> &a, Vector <complexd> &w,
    int proc_Row, int proc_Col, int numQBits, int Emin, int Emax, int globalN, int localN, int proc_M, int proc_N)
{
    for (int i = 0; i < proc_M; ++i) {
        for (int j = 0; j < proc_N; ++j) {
            int globalI = basis.data[localN * proc_Row + i];
            int globalJ = basis.data[localN * proc_Col + j];
            if (globalI == globalJ) {
                H.data[i * proc_N + j] = sumActiveQBits(w.data, numQBits, globalI);
            }
            else if ((areNeighbours(globalI, globalJ)) && (Energy(globalI) == Energy(globalJ))) {
                int mask = globalI ^ globalJ;
                H.data[i * proc_N + j] = a.data[numQBits - (int) log2(mask / 3) - 2];
                // if (commRank == 1) cout << globalI << " " << globalJ << endl; // DBG
            }
        }
    }
}

void initRo(Matrix_complexd &ro, Vector <complexd> &phi,
    int proc_Row, int proc_Col, int numQBits, int Emin, int Emax, int globalN, int localN, int proc_M, int proc_N)
{
  for (int i = 0; i < proc_M; ++i) {
    for (int j = 0; j < proc_N; ++j) {
      int globalI = basis.data[localN * proc_Row + i];
      int globalJ = basis.data[localN * proc_Col + j];

      ro.data[i * proc_N + j] = phi.data[globalI] * conj(phi.data[globalJ]);
    }
  }
}
