CC=nvcc 
CFLAGS+= -g -G -std=c++17 
LIBS= -lnccl -lmpi

OBJS0= example_3
OBJS1= example_3alltoall
OBJS2= example_3gather
OBJS3= example_3scatter

all: ${OBJS0}.out ${OBJS1}.out ${OBJS2}.out ${OBJS3}.out


${OBJS0}.out: ${OBJS0}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS}

${OBJS1}.out: ${OBJS1}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS} -I../

${OBJS2}.out: ${OBJS2}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS} -I../

${OBJS3}.out: ${OBJS3}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS} -I../

clean: 
	${RM} *.out -r

test:
	mpirun -np 2 ${OBJS0}.out 
	mpirun -np 2 ${OBJS1}.out 
	mpirun -np 2 ${OBJS2}.out 
	mpirun -np 2 ${OBJS3}.out 