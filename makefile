all: bench

bench: bench.cpp threefry_avx.h
	c++ -o $@ -march=native -mtune=native -mavx2 -mfma -O3 -std=c++17 $< --save-temps # -fno-tree-vectorize

clean:
	rm -f bench
