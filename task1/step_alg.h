#include <mpi.h>
#include <cstdlib>
#include <iostream>
#include <complex>
using namespace std;

typedef complex <double> complexd;

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
    void descinit_(int *descArray, int *M, int *N, int *Mb, int *Nb, int *dummy1 , int *dummy2 , int *context, int *gridRows, int *info);
    void pzheevd_(char *jobZ, char *upLo, int *N, complexd *A, int *ia, int *ja, int *descA, complexd *w,
            		complexd *Z, int *iz, int *jz, int *descZ, complexd *work, int *lwork, complexd *rwork, int *lrwork, complexd *iwork, int *liwork, int *info);
    void pzgemm_(char *transA, char *transB, int *M, int *N, int *K, complexd *alpha, complexd *A, int *ia, int *ja, int *descA,
            		complexd *B, int *ib, int *jb, int *descB, complexd *beta, complexd *C, int *ic, int *jc, int *descC);
    void pzgetrf_(int *N, int *M, complexd *A, int *ia, int *ja, int *descA, int *iPiv, int *info);
    void pzgetri_(int *N, complexd *A, int *ia, int *ja, int *descA, int *iPiv, complexd *work, int *lwork, complexd *iwork, int *liwork, int *info);
}

class Matrix_complexd {
public:
    
	complexd *data;
    int desc[9];
    int lld;
    
	Matrix_complexd(int proc_M = 2, int proc_N = 2, int globalN = 4, int localN = 2, int context = 0) {
        data = new complexd[proc_M * proc_N];
        if (data == NULL) {
  		    cerr << "Error on matrix allocation" << endl;
     		MPI_Finalize();
      		exit(-1);
    	}
    	lld = proc_M > 1 ? proc_M : 1;
    	int zero = 0, info = 0;
    	descinit_(desc, &globalN, &globalN, &localN, &localN, &zero, &zero, &context, &lld, &info);
    	if (info != 0) {
      		cerr << "Error on descinit_" << endl;
      		Cblacs_gridexit(context);
      		MPI_Finalize();
      		exit(-1);
    	}
  	}

    void readMatrix(int globalN, int localN, int proc_M, int proc_N, int proc_Row, int proc_Col, string fileName) {
    	MPI_File mpiFile;
    	MPI_File_open(MPI_COMM_WORLD, "H", MPI_MODE_RDONLY, MPI_INFO_NULL, &mpiFile);
    	MPI_File_set_view(mpiFile, localN * proc_Col + localN * globalN * proc_Row, MPI_DOUBLE_COMPLEX, MPI_DOUBLE_COMPLEX, "native", MPI_INFO_NULL);
    	MPI_File_read_ordered(mpiFile, data, proc_M * proc_N, MPI_DOUBLE_COMPLEX, MPI_STATUS_IGNORE);
    	MPI_File_close(&mpiFile);
	}
    
	~Matrix_complexd() {
    	delete[] data;
  	}
};

class Vector_complexd {
public:
    
	complexd *data;
    
	Vector_complexd(int N = 10) {
    	data = new complexd[N];
    	if (data == NULL) {
      		cerr << "Error on Vector_complexd allocation" << endl;
      	MPI_Finalize();
      	exit(-1);
    	}
  	}
  	~Vector_complexd() {
    	delete[] data;
  	}
};

class Vector_int {
public:
  	
	int *data;
  	
	Vector_int(int N = 10) {
    	data = new int[N];
    	if (data == NULL) {
      		cerr << "Error on Vector_complexd allocation" << endl;
      		MPI_Finalize();
      		exit(-1);
    	}
  	}
  
  	~Vector_int() {
    	delete[] data;
  	}
};

void step_alg(Matrix_complexd &H, Matrix_complexd &ro, int globalN, int localN, int proc_M, int proc_N, 
                                                                int context, int proc_Row, int proc_Col, 
                                                                int commRank, int commSize, 
                                                                int Rows, int Cols, int n);
