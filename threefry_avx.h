#include "kernel_helpers.h"

#include <immintrin.h>



void printvec(const __m256i v, auto str){
  std::uint32_t x [8];
  _mm256_storeu_si256((__m256i*) &x[0], v);
  std::cout << str << ": " << x[0] << std::endl;
}

void ThreeFry2x32Kernel(const std::uint32_t* key0,
                                   const std::uint32_t* key1,
                                   const std::uint32_t* data0,
                                   const std::uint32_t* data1,
                                   std::uint32_t* out0,
                                   std::uint32_t* out1,
                                   std::int64_t n) {


  constexpr std::uint8_t rotations0[4] = {13, 15, 26, 6};
  constexpr std::uint8_t rotations1[4] = {17, 29, 16, 24};

  constexpr std::uint32_t _ks2 = 0x1BD11BDA;

  const __m256i ks0 = _mm256_set1_epi32(*key0);
  const __m256i ks1 = _mm256_set1_epi32(*key1);
  const __m256i ks2 = _mm256_set1_epi32( _ks2^(*key0)^(*key1) );

  // assume n is multiple of 8 for now; TODO more unrolling
  for (std::int64_t idx = 0; idx < n; idx += 8) {

    __m256i x0 = _mm256_loadu_si256((__m256i*) &data0[idx]);
    __m256i x1 = _mm256_loadu_si256((__m256i*) &data1[idx]);

    auto rotate_left = [](const __m256i v, const std::uint8_t distance) {

      const __m256i v1 = _mm256_slli_epi32(v, distance); // lat 1, tp 2
      const __m256i v2 = _mm256_srli_epi32(v, 32-distance);
      const __m256i v3 = _mm256_or_si256(v1, v2); // lat 1, tp 3
      return v3;
    };

    auto round = [&rotate_left](const __m256i v0, const __m256i v1, const std::uint8_t rotation) {
      const __m256i v0a = _mm256_add_epi32(v0, v1); // lat 1, tp 3
      const __m256i v1a = rotate_left(v1, rotation);
      const __m256i v1b = _mm256_xor_si256(v1a, v0a); // lat 1, tp 3
      return std::make_pair(v0a, v1b);
    };

    x0 = _mm256_add_epi32(x0,ks0); // lat 1, tp 3
    x1 = _mm256_add_epi32(x1,ks1); // lat 1, tp 3

    for (const auto& r: rotations0) {
      std::tie(x0, x1) = round(x0, x1, r);
    }

    const __m256i one = _mm256_set1_epi32(1u);
    const __m256i ks2p1 = _mm256_add_epi32(ks2, one);
    x0 = _mm256_add_epi32(x0,ks1);
    x1 = _mm256_add_epi32(x1,ks2p1);

    for (const auto& r: rotations1) {
      std::tie(x0, x1) = round(x0, x1, r);
    }

    const __m256i two = _mm256_add_epi32(one, one);
    const __m256i ks0p2 = _mm256_add_epi32(ks0, two);
    x0 = _mm256_add_epi32(x0,ks2);
    x1 = _mm256_add_epi32(x1, ks0p2);

    for (const auto& r: rotations0) {
      std::tie(x0, x1) = round(x0, x1, r);
    }

    const __m256i three = _mm256_add_epi32(two, one);
    const __m256i ks1p3 = _mm256_add_epi32(ks1, three);
    x0 = _mm256_add_epi32(x0,ks0);
    x1 = _mm256_add_epi32(x1, ks1p3);

    for (const auto& r: rotations1) {
      std::tie(x0, x1) = round(x0, x1, r);
    }

    const __m256i four = _mm256_add_epi32(three, one);
    const __m256i ks2p4 = _mm256_add_epi32(ks2, four);
    x0 = _mm256_add_epi32(x0,ks1);
    x1 = _mm256_add_epi32(x1,ks2p4);

    for (const auto& r: rotations0) {
      std::tie(x0, x1) = round(x0, x1, r);
    }

    const __m256i five = _mm256_add_epi32(four, one);
    const __m256i ks0p5 = _mm256_add_epi32(ks0, five);
    __m256i y0 = _mm256_add_epi32(x0,ks2);
    __m256i y1 =  _mm256_add_epi32(x1, ks0p5);

    _mm256_storeu_si256((__m256i*) &out0[idx], y0);
    _mm256_storeu_si256((__m256i*) &out1[idx], y1);

  }
}




void ThreeFry2x32Kernel0(const std::uint32_t* key0,
                                   const std::uint32_t* key1,
                                   const std::uint32_t* data0,
                                   const std::uint32_t* data1,
                                   std::uint32_t* out0,
                                   std::uint32_t* out1,
                                   std::int64_t n) {

std::uint32_t ks[3];

// 0x1BD11BDA is a parity constant specified by the ThreeFry2x32 algorithm.
ks[0] = key0[0];
ks[1] = key1[0];

std::cout << "keys  " << ks[0] << " " << ks[1] << std::endl;

ks[2] = 0x1BD11BDA;
ks[2] = ks[2] ^ key0[0];
ks[2] = ks[2] ^ key1[0];

const std::uint32_t rotations[8] = {13, 15, 26, 6, 17, 29, 16, 24};

for (std::int64_t idx = 0; idx < n;  idx += 1) {
    // Rotation distances specified by the Threefry2x32 algorithm.
    std::uint32_t x[2];
    x[0] = data0[idx];
    x[1] = data1[idx];

    auto rotate_left = [](std::uint32_t v, std::uint32_t distance) {
      return (v << distance) | (v >> (32 - distance));
    };

    // Performs a single round of the Threefry2x32 algorithm, with a rotation
    // amount 'rotation'.
    auto round = [&](std::uint32_t* v, std::uint32_t rotation) {
      v[0] += v[1];
      v[1] = rotate_left(v[1], rotation);
      v[1] ^= v[0];
    };

    // There are no known statistical flaws with 13 rounds of Threefry2x32.
    // We are conservative and use 20 rounds.
    x[0] = x[0] + ks[0];
    x[1] = x[1] + ks[1];
    for (int i = 0; i < 4; ++i) {
      round(x, rotations[i]);
    }

    x[0] = x[0] + ks[1];
    x[1] = x[1] + ks[2] + 1u;
    for (int i = 4; i < 8; ++i) {
      round(x, rotations[i]);
    }

    x[0] = x[0] + ks[2];
    x[1] = x[1] + ks[0] + 2u;
    for (int i = 0; i < 4; ++i) {
      round(x, rotations[i]);
    }

    x[0] = x[0] + ks[0];
    x[1] = x[1] + ks[1] + 3u;
    for (int i = 4; i < 8; ++i) {
      round(x, rotations[i]);
    }

    x[0] = x[0] + ks[1];
    x[1] = x[1] + ks[2] + 4u;
    for (int i = 0; i < 4; ++i) {
      round(x, rotations[i]);
    }

    out0[idx] = x[0] + ks[2];
    out1[idx] = x[1] + ks[0] + 5u;
  }

}

#include<iostream>

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

  ThreeFry2x32Kernel(keys[0], keys[1], data[0], data[1], out[0], out[1], n);
}
