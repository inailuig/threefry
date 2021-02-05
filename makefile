all: bench

bench: bench.cpp
	c++ -o $@ -march=native -mtune=native -mavx2 -mfma -O3 -std=c++17 $< # -fno-tree-vectorize

clean:
	rm -f bench
