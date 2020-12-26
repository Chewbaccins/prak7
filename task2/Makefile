OPT=	-O2
FLAGS=	-DUSE_COMPLEX -DUSE_DOUBLE

LIBPATH=`pwd`/../scalapack-2.1.0/
LIBS=	$(LIBPATH)libscalapack.a \
	-lpthread -lblas -llapack -lm -lgfortran

all: matrix_vector step_alg main matrix_vector.cpp step_alg.cpp main.cpp

matrix_vector: matrix_vector.cpp Makefile
	mpicxx -c -o $@ $(OPT) matrix_vector.cpp $(LIBS)

step_alg: step_alg.cpp Makefile
	mpicxx -c -o $@ $(OPT) step_alg.cpp $(LIBS)

main: main.cpp step_alg.cpp matrix_vector.cpp Makefile
	mpicxx -o $@ $(OPT) main.cpp step_alg matrix_vector $(LIBS)
