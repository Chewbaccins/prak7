# 1 task

423 Тыркалов Евгений, Ни Юлия.

###### compile
make all

###### run
mpirun -np [number of processes] ./main [global matrix size] [local matrix size] [number of steps]

###### example
mpirun -np 4 ./main 16 8 6
