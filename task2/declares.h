#include <iomanip>
#include <fstream>
#include <cassert>
typedef complex<double> complexd;

extern int commRank, commSize;

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
            complexd *Z, int *iz, int *jz, int *descZ, double *work, int *lwork, complexd *rwork, int *lrwork, int *iwork, int *liwork, int *info);
    void pzgemm_(char *transA, char *transB, int *M, int *N, int *K, complexd *alpha, complexd *A, int *ia, int *ja, int *descA,
            complexd *B, int *ib, int *jb, int *descB, complexd *beta, complexd *C, int *ic, int *jc, int *descC);
}

template <typename dataType> class Vector {
public:
    dataType *data;
    int length;

    Vector(int N = 10) {
        data = new dataType[N] {(dataType) 0};
        if (data == NULL) {
        cerr << "Error on vector allocation" << endl;
        MPI_Finalize();
        exit(-1);
        }
        length = N;
    }

    void resize(int N) {
        delete[] data;
        data = new dataType[N] {(dataType) 0};
        if (data == NULL) {
        cerr << "Error on vector allocation" << endl;
        MPI_Finalize();
        exit(-1);
        }
        length = N;
    }

    void read(string fileName) {
        fstream f(fileName);
        assert(f.is_open());
        for (int i = 0; i < length; ++i) {
        if (!f.eof()) f >> data[i];
        if (f.eof()) {
            cerr << "No enough values in input file" << endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
            exit(-1);
        }
        }
        f.close();
    }

    void normal() {
        double norm = 0.0d;
        for (int i = 0; i < length; ++i) {
            norm += data[i].real() * data[i].real() + data[i].imag() * data[i].imag();
        }
        norm = sqrt(norm);
        for (int i = 0; i < length; ++i) {
            data[i] /= norm;
        }
    }

    ~Vector() {
        delete[] data;
    }
};

class Matrix_complexd {
public:
    complexd *data;
    int desc[9];
    int lld;

    Matrix_complexd(int proc_M = 2, int proc_N = 2, int globalN = 4, int localN = 2, int context = 0) {
        data = new complexd[proc_M * proc_N] {0};
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

    void gather(int Rows, int Cols, int proc_Row, int proc_Col, int globalN, int localN) { // void -> vector, gather -> gatherComplexD
        Vector <complexd> gatheredMatrix; // not quite correct
        Vector <int> displacements(commSize);
        Vector <int> counts(commSize);
        int zero = 0, displacement = 0;

        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                int m = numroc_(&globalN, &localN, &i, &zero, &Rows);
                int n = numroc_(&globalN, &localN, &j, &zero, &Cols);
                counts.data[i * Cols + j] = m * n;
                displacements.data[i * Cols + j] = displacement;
                displacement += m * n;
            }
        }

        int proc_M = numroc_(&globalN, &localN, &proc_Row, &zero, &Rows);
        int proc_N = numroc_(&globalN, &localN, &proc_Col, &zero, &Cols);
        if (commRank == 0) {
            gatheredMatrix.resize(globalN * globalN);
        }
        MPI_Gatherv(data, proc_M * proc_N, MPI_DOUBLE_COMPLEX, gatheredMatrix.data, counts.data, displacements.data, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
        if (commRank == 0) {
            for (int row = 0; row < Rows; ++row) {
                int m = numroc_(&globalN, &localN, &row, &zero, &Rows);

                for (int i = 0; i < m; ++i) {
                for (int col = 0; col < Cols; ++col) {
                    int offset = displacements.data[row * Cols + col];
                    int n = numroc_(&globalN, &localN, &col, &zero, &Cols);

                    for (int j = 0; j < n; ++j)
                    cout << setw(23) << gatheredMatrix.data[offset + i * n + j];
                }
                cout << endl;
                }
            }
        }
    }

    ~Matrix_complexd() {
        delete[] data;
    }
};
