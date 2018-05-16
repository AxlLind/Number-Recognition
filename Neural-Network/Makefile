.SILENT:

CC=clang++
FLAGS=-std=c++1z -O3 -Wall -g
OUT=a.out

compile:
	$(CC) $(FLAGS) -o $(OUT) main.cpp

r:
	./$(OUT)

valgrind:
	valgrind --tool=memcheck --leak-check=yes ./$(OUT)
