#include "threefry_avx.h"
#include "tsc_x86.h"
#include <numeric>
#include<iostream>

int main(){
  std::uint32_t keys[] = {0, 123};

  constexpr std::uint64_t n_samples = 3*(2<<10);
  constexpr std::uint64_t n = n_samples/2;

  std::uint32_t* data = new std::uint32_t[n_samples];
  std::iota(data, data+n_samples, 0);

  std::uint32_t* data0 = data;
  std::uint32_t* data1 = data+n;

  std::uint32_t* out = new std::uint32_t[n_samples];
  //std::uint32_t* out = data;
  std::uint32_t* out0 = out;
  std::uint32_t* out1 = out+n;

  std::fill(out, out+n_samples, 0);
  ThreeFry2x32Kernel0(&keys[0], &keys[1], data0, data1, out0, out1, n);
  std::cout << "d0: "  << out[0] << std::endl;

  std::fill(out, out+n_samples, 0);
  ThreeFry2x32Kernel1(&keys[0], &keys[1], data0, data1, out0, out1, n);
  std::cout << "d1: "  << out[0] << std::endl;

  std::fill(out, out+n_samples, 0);
  ThreeFry2x32Kernel2(&keys[0], &keys[1], data0, data1, out0, out1, n);
  std::cout << "d2: "  << out[0] << std::endl;

  std::fill(out, out+n_samples, 0);
  ThreeFry2x32Kernel3(&keys[0], &keys[1], data0, data1, out0, out1, n);
  std::cout << "d3: "  << out[0] << std::endl;

  std::fill(out, out+n_samples, 0);
  ThreeFry2x32Kernel4(&keys[0], &keys[1], data0, data1, out0, out1, n);
  std::cout << "d4: "  << out[0] << std::endl;

  myInt64 start, end;
	double cycles = 0.;
  double perf = 0.0;
  long num_runs = 16384;
  //long flops = 117*n; // (4*5+3)*5 + 2
  //long flops = 37*n; // (4*1+3)*5 + 2; add only
  long flops = 77*n; // (4*3+3)*5 + 2; add & logic ops only, shift not

  start = start_tsc();
  for (size_t i = 0; i < num_runs; ++i) {
    ThreeFry2x32Kernel0(&keys[0], &keys[1], data0, data1, out0, out1, n);
  }
  end = stop_tsc(start);
  cycles = ((double)end) / num_runs;
  perf = flops / cycles;

  std::cout << "cycles0 " <<  cycles << std::endl;
  std::cout << "perf0 " <<  perf << std::endl;




  start = start_tsc();
  for (size_t i = 0; i < num_runs; ++i) {
    ThreeFry2x32Kernel1(&keys[0], &keys[1], data0, data1, out0, out1, n);
  }
  end = stop_tsc(start);
  cycles = ((double)end) / num_runs;
  perf = flops / cycles;

  std::cout << "cycles1 " <<  cycles << std::endl;
  std::cout << "perf1 " <<  perf << std::endl;



  start = start_tsc();
  for (size_t i = 0; i < num_runs; ++i) {
    ThreeFry2x32Kernel2(&keys[0], &keys[1], data0, data1, out0, out1, n);
  }
  end = stop_tsc(start);
  cycles = ((double)end) / num_runs;
  perf = flops / cycles;

  std::cout << "cycles2 " <<  cycles << std::endl;
  std::cout << "perf2 " <<  perf << std::endl;


  start = start_tsc();
  for (size_t i = 0; i < num_runs; ++i) {
    ThreeFry2x32Kernel3(&keys[0], &keys[1], data0, data1, out0, out1, n);
  }
  end = stop_tsc(start);
  cycles = ((double)end) / num_runs;
  perf = flops / cycles;

  std::cout << "cycles3 " <<  cycles << std::endl;
  std::cout << "perf3 " <<  perf << std::endl;

  start = start_tsc();
  for (size_t i = 0; i < num_runs; ++i) {
    ThreeFry2x32Kernel4(&keys[0], &keys[1], data0, data1, out0, out1, n);
  }
  end = stop_tsc(start);
  cycles = ((double)end) / num_runs;
  perf = flops / cycles;

  std::cout << "cycles4 " <<  cycles << std::endl;
  std::cout << "perf4 " <<  perf << std::endl;

  delete[] data;
  delete[] out;

}
