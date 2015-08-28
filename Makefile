all: program

program: erosionFuncTemplate.o erosion.o erosionCPU.o
	g++ -o program -I./common/inc -I/usr/local/cuda/include -lcuda main.cpp  erosionFuncTemplate.o erosion.o erosionCPU.o -L/usr/local/cuda/lib64 -lcudart -std=c++11

erosionFuncTemplate.o:
	nvcc -I./common/inc -c -arch=sm_20 erosionFuncTemplate.cu -L/usr/local/cuda/lib64 -lcudart -gencode arch=compute_20,code=sm_20

erosion.o:
	nvcc -I./common/inc -c -arch=sm_20 erosion.cu -L/usr/local/cuda/lib64 -lcudart -gencode arch=compute_20,code=sm_20

erosionCPU.o: erosionCPU.cpp
	g++ -c erosionCPU.cpp

clean: 
	rm -rf *o program