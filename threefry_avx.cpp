#include "threefry_avx.h"
#include "kernel_helpers.h"
#include "kernel_pybind11_helpers.h"
#include "pybind11/pybind11.h"


void ThreeFry2x32(void** outbuf, void **inbuf) {

  const std::int64_t n = *reinterpret_cast<const std::int64_t *>(inbuf[0]);

  std::array<const std::uint32_t*, 2> keys;
  keys[0] = reinterpret_cast<const std::uint32_t*>(inbuf[1]);
  keys[1] = reinterpret_cast<const std::uint32_t*>(inbuf[2]);

  std::array<const std::uint32_t*, 2> data;
  data[0] = reinterpret_cast<const std::uint32_t*>(inbuf[3]);
  data[1] = reinterpret_cast<const std::uint32_t*>(inbuf[4]);

  std::array<std::uint32_t*, 2> out;
  out[0] = reinterpret_cast<std::uint32_t*>(outbuf[0]);
  out[1] = reinterpret_cast<std::uint32_t*>(outbuf[1]);

  //std::cout << "n: " << n << std::endl;

  ThreeFry2x32Kernel2(keys[0], keys[1], data[0], data[1], out[0], out[1], n);
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["threefry2x32"] = EncapsulateFunction(ThreeFry2x32);
  return dict;
}

PYBIND11_MODULE(threefry_avx, m) {
  m.def("registrations", &Registrations);
}
