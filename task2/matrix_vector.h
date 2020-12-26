extern int commRank, commSize;
extern Vector <int> basis;

int C(int n, int k);
string binary_output(int numQBits, int classicState);
int Energy(int classicState);
complexd sumActiveQBits(int classicState, int globalN, complexd *wData);
bool areNeighbours(int classicState1, int classicState2, int globalN);
void initH(Matrix_complexd &H, Vector <complexd> &a, Vector <complexd> &w,
    int proc_Row, int proc_Col, int numQBits, int Emin, int Emax, int globalN, int localN, int proc_M, int proc_N);
void initRo(Matrix_complexd &ro, Vector <complexd> &phi, int proc_Row, int proc_Col, int numQBits, int Emin, int Emax, int globalN, int localN, int proc_M, int proc_N);
