OPT=	-O2
FLAGS=	-DUSE_COMPLEX -DUSE_DOUBLE

LIBPATH=`pwd`/../scalapack-2.1.0/
LIBS=	$(LIBPATH)libscalapack.a \
	-lpthread -lblas -llapack -lm -lgfortran

all: step_alg main step_alg.cpp main.cpp

step_alg: step_alg.cpp Makefile
	mpicxx -c -o $@ $(OPT) step_alg.cpp $(LIBS)

main: main.cpp step_alg.cpp Makefile
	mpicxx -o $@ $(OPT) main.cpp step_alg $(LIBS)

init: initFiles.cpp
	g++ -o $@ initFiles.cpp

view: view_file.cpp
	g++ -o $@ view_file.cpp