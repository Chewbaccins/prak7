# 1 task

###### compile
make all

###### run
mpirun -np [number of processes] ./main [global matrix size] [local matrix size] [number of steps]

###### example
mpirun -np 4 ./main 16 8 6