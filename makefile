all: bench

bench: bench.cpp threefry_avx.h
	c++ -o $@ -march=native -mtune=native -mavx2 -mfma -O3 -std=c++17 $< --save-temps # -fno-tree-vectorize

bench_gcc: bench.cpp threefry_avx.h
	g++-10 -o $@ -march=native -mtune=native -mavx2 -mfma -O3 -std=c++17 $< --save-temps -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk

bench_icc: bench.cpp threefry_avx.h
	icc -o $@ -xHost -mavx -O2 -std=c++17 $< --save-temps -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk

clean:
	rm -f bench bench.ii bench.o bench.s bench.bc
