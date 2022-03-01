CC=nvcc 
CFLAGS+= -g -G -std=c++17
LIBS= -lnccl -lmpi

OBJS0= example_2allreduce
OBJS1= example_2brdcast
OBJS2= example_2gather
OBJS3= example_2reduce
OBJS4= example_2reduceScatter
OBJS5= example_2sendRecv
OBJS6= example_2sendRecvbind

all: ${OBJS0}.out ${OBJS1}.out ${OBJS2}.out ${OBJS3}.out ${OBJS4}.out ${OBJS5}.out ${OBJS6}.out 

${OBJS0}.out: ${OBJS0}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS}

${OBJS1}.out: ${OBJS1}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS} -I../

${OBJS2}.out: ${OBJS2}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS} -I../

${OBJS3}.out: ${OBJS3}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS} -I../

${OBJS4}.out: ${OBJS4}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS} -I../

${OBJS5}.out: ${OBJS5}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS} -I../

${OBJS6}.out: ${OBJS6}.cu
	${CC} ${CFLAGS} $^ -o $@ ${LIBS} -I../

clean: 
	${RM} *.out -r

test:
	mpirun -np 2 ${OBJS0}.out 1 
	mpirun -np 2 ${OBJS1}.out 1
	mpirun -np 2 ${OBJS2}.out 1
	mpirun -np 2 ${OBJS3}.out 1
	mpirun -np 2 ${OBJS4}.out 1
	mpirun -np 2 ${OBJS5}.out 1
	mpirun -np 2 ${OBJS6}.out 1