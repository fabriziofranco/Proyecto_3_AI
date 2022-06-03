all: run clean

run:
	g++ mlp.cpp -std=c++17 -o mlp.out -larmadillo
	./mlp.out

clean:
	rm -f *.bin *.txt *.blif *.out *~

.PHONY: all clean