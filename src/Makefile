all: run clean P

run:
	g++ mlp.cpp -std=c++17 -o mlp.out -larmadillo
	./mlp.out

clean:
	rm -f *.bin *.txt *.blif *.out *~

P:
	g++ parser.cpp -std=c++17 -o parser.out -lstdc++fs
	./parser.out

.PHONY: all clean