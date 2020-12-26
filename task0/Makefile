OPT=	-O2
FLAGS=	-DUSE_COMPLEX -DUSE_DOUBLE

LIBPATH=`pwd`/../scalapack-2.1.0/
LIBS=	$(LIBPATH)libscalapack.a \
	-lpthread -lblas -llapack -lm -lgfortran

all: Matrix_multiply

Matrix_multiply: Matrix_multiply.cpp Makefile
	mpicxx -o $@ $(OPT) Matrix_multiply.cpp $(LIBS)