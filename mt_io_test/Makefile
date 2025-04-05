#
# Builds the tester.
#

EXE=mt_io

main.o:
	g++ -c main.o -Wall -O3 main.cpp
	
$(EXE): main.o
	g++ -o $(EXE) main.o

clean:
	rm $(EXE) main.o
