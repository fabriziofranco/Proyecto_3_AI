all: run clean P

run:
	g++ -o mlp.out mlp.cpp -std=c++17 -larmadillo -llapack -lstdc++fs -fopenmp
	./mlp.out

clean:
	rm -f *.bin *.txt *.blif *.out *~

P:
	g++ parser.cpp -std=c++17 -o parser.out -lstdc++fs
	./parser.out

.PHONY: all clean