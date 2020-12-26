# 2 task

423 Тыркалов Евгений, Ни Юлия. Вариант 1.

###### compile
make all

###### run
mpirun -np [number of processes] ./main [num qbits] [num steps] [min excited atoms] [max excited atoms]

###### example
mpirun -np 4 ./main 3 3 1 3

Vectors a, w, phi are read from files. phi is normalized after read. It is available to modify them in "Build" folder.

If length of basis is > 8, output goes to "Build/output.txt"
