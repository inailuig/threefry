#include "threefry_avx.h"

#include <numeric>
#include<iostream>
int main(){
  std::uint32_t keys[] = {0, 123};

  constexpr std::uint32_t n_samples = 16;
  constexpr std::uint32_t n = n_samples/2;

  std::uint32_t* data = new std::uint32_t[n_samples];

  std::iota(data, data+n_samples, 0);

  std::uint32_t* data0 = data;
  std::uint32_t* data1 = data+n;
  //std::uint32_t* out = new std::uint32_t[n_samples];
  std::uint32_t* out = data;
  std::uint32_t* out0 = data;
  std::uint32_t* out1 = data+n;

  ThreeFry2x32Kernel0(&keys[0], &keys[1], data0, data1, out0, out1, n);
  std::cout << "d0: "  << data[0] << std::endl;

  std::iota(data, data+n_samples, 0);

  ThreeFry2x32Kernel(&keys[0], &keys[1], data0, data1, out0, out1, n);
  std::cout << "d1: "  << data[0] << std::endl;


  delete[] data;

}
