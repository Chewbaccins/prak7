# 0 task PDGEMM/PZGEMM

###### compile
make matrix_multiply

###### run
mpirun -np [number of processes] ./matrix_multiply [global matrix size] [local size] [value type]

value types:

z - complex values

d - real values

###### example
mpirun -np 4 ./matrix_multiply 100 50 z
