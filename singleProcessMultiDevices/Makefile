CC=nvcc 
CFLAGS+= -g -G  -std=c++17
LIBS= -lnccl -lmpi

OBJS0=example_1
OBJS1=example_1reduceScatter
OBJS2=example_1sendRecv
OBJS3=example_1sendRecvbind

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
	./${OBJS0}.out 
	./${OBJS1}.out 
	./${OBJS2}.out 
	./${OBJS3}.out 
