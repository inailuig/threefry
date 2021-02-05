#include "threefry_avx.h"

#include "kernel_pybind11_helpers.h"
#include "pybind11/pybind11.h"

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["threefry2x32"] = EncapsulateFunction(ThreeFry2x32);
  return dict;
}

PYBIND11_MODULE(threefry_avx, m) {
  m.def("registrations", &Registrations);
}
