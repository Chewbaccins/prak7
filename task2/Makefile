OPT=	-O2
FLAGS=	-DUSE_COMPLEX -DUSE_DOUBLE

LIBPATH=`pwd`/../scalapack-2.1.0/
LIBS=	$(LIBPATH)libscalapack.a \
	-lpthread -lblas -llapack -lm -lgfortran

all: Ro_H step_alg main Ro_H.cpp step_alg.cpp main.cpp

Ro_H: Ro_H.cpp Makefile
	mpicxx -c -o $@ $(OPT) Ro_H.cpp $(LIBS)

step_alg: step_alg.cpp Makefile
	mpicxx -c -o $@ $(OPT) step_alg.cpp $(LIBS)

main: main.cpp step_alg.cpp Ro_H.cpp Makefile
	mpicxx -o $@ $(OPT) main.cpp step_alg Ro_H $(LIBS)
