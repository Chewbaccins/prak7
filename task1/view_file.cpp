#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <complex>
#include <vector>
using namespace std;

typedef complex <double> complexd;

int main(int argc, char **argv)
{

    if (argc < 2) {
		cerr << "Invalid command line. Usage: /view fileName n" << endl;
		exit(-1);
	}

    ifstream ifs(argv[1], ios::binary | ios::in);
    int n = stoi(argv[2]);
    vector<complexd> matrix(n*n);
    ifs.read(reinterpret_cast<char*>(matrix.data()), 2*n*n*sizeof(double));
    ifs.close();

	for (int i = 0; i < n*n; i++) {
		cout << matrix[i] << " ";
		if ((i+1) % n == 0) cout << endl;
	}

    return 0;
}