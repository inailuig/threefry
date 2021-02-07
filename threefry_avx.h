#include <immintrin.h>
#include <tuple>
#include <cstdint>


// TODO aligned load/store?

void ThreeFry2x32Kernel4(const std::uint32_t* key0,
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

  for (std::int64_t idx = 0; idx < n; idx += 32) {

    __m256i x0_0 = _mm256_loadu_si256((__m256i*) &data0[idx]);
    __m256i x0_1 = _mm256_loadu_si256((__m256i*) &data0[idx+8]);
    __m256i x0_2 = _mm256_loadu_si256((__m256i*) &data0[idx+16]);
    __m256i x0_3 = _mm256_loadu_si256((__m256i*) &data0[idx+24]);


    __m256i x1_0 = _mm256_loadu_si256((__m256i*) &data1[idx]);
    __m256i x1_1 = _mm256_loadu_si256((__m256i*) &data1[idx+8]);
    __m256i x1_2 = _mm256_loadu_si256((__m256i*) &data1[idx+16]);
    __m256i x1_3 = _mm256_loadu_si256((__m256i*) &data1[idx+24]);


    auto round = [](const __m256i v0_0, const __m256i v1_0, const __m256i v0_1, const __m256i v1_1, const __m256i v0_2, const __m256i v1_2, const __m256i v0_3, const __m256i v1_3, const std::uint8_t rotation) {
      const __m256i v0a_0 = _mm256_add_epi32(v0_0, v1_0); // lat 1, tp 3
      const __m256i v0a_1 = _mm256_add_epi32(v0_1, v1_1); // lat 1, tp 3
      const __m256i v0a_2 = _mm256_add_epi32(v0_2, v1_2); // lat 1, tp 3
      const __m256i v0a_3 = _mm256_add_epi32(v0_3, v1_3); // lat 1, tp 3

      const __m256i w1_0 = _mm256_slli_epi32(v1_0, rotation); // lat 1, tp 2
      const __m256i w1_1 = _mm256_slli_epi32(v1_1, rotation); // lat 1, tp 2
      const __m256i w1_2 = _mm256_slli_epi32(v1_2, rotation); // lat 1, tp 2
      const __m256i w1_3 = _mm256_slli_epi32(v1_3, rotation); // lat 1, tp 2

      const __m256i w2_0 = _mm256_srli_epi32(v1_0, 32-rotation);
      const __m256i w2_1 = _mm256_srli_epi32(v1_1, 32-rotation);
      const __m256i w2_2 = _mm256_srli_epi32(v1_2, 32-rotation);
      const __m256i w2_3 = _mm256_srli_epi32(v1_3, 32-rotation);

      const __m256i v1a_0 = _mm256_or_si256(w1_0, w2_0); // lat 1, tp 3
      const __m256i v1a_1 = _mm256_or_si256(w1_1, w2_1); // lat 1, tp 3
      const __m256i v1a_2 = _mm256_or_si256(w1_2, w2_2); // lat 1, tp 3
      const __m256i v1a_3 = _mm256_or_si256(w1_3, w2_3); // lat 1, tp 3

      const __m256i v1b_0 = _mm256_xor_si256(v1a_0, v0a_0); // lat 1, tp 3
      const __m256i v1b_1 = _mm256_xor_si256(v1a_1, v0a_1); // lat 1, tp 3
      const __m256i v1b_2 = _mm256_xor_si256(v1a_2, v0a_2); // lat 1, tp 3
      const __m256i v1b_3 = _mm256_xor_si256(v1a_3, v0a_3); // lat 1, tp 3

      return std::make_tuple(v0a_0, v1b_0, v0a_1, v1b_1,v0a_2, v1b_2, v0a_3, v1b_3);
    };

    x0_0 = _mm256_add_epi32(x0_0,ks0); // lat 1, tp 3
    x0_1 = _mm256_add_epi32(x0_1,ks0); // lat 1, tp 3
    x0_2 = _mm256_add_epi32(x0_2,ks0); // lat 1, tp 3
    x0_3 = _mm256_add_epi32(x0_3,ks0); // lat 1, tp 3

    x1_0 = _mm256_add_epi32(x1_0,ks1); // lat 1, tp 3
    x1_1 = _mm256_add_epi32(x1_1,ks1); // lat 1, tp 3
    x1_2 = _mm256_add_epi32(x1_2,ks1); // lat 1, tp 3
    x1_3 = _mm256_add_epi32(x1_3,ks1); // lat 1, tp 3

    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2, x0_3, x1_3) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, x0_3, x1_3, rotations0[0]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2, x0_3, x1_3) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, x0_3, x1_3, rotations0[1]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2, x0_3, x1_3) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, x0_3, x1_3, rotations0[2]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2, x0_3, x1_3) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, x0_3, x1_3, rotations0[3]);

    const __m256i one = _mm256_set1_epi32(1u);
    const __m256i ks2p1 = _mm256_add_epi32(ks2, one);

    x0_0 = _mm256_add_epi32(x0_0,ks1);
    x0_1 = _mm256_add_epi32(x0_1,ks1);
    x0_2 = _mm256_add_epi32(x0_2,ks1);
    x0_3 = _mm256_add_epi32(x0_3,ks1);

    x1_0 = _mm256_add_epi32(x1_0,ks2p1);
    x1_1 = _mm256_add_epi32(x1_1,ks2p1);
    x1_2 = _mm256_add_epi32(x1_2,ks2p1);
    x1_3 = _mm256_add_epi32(x1_3,ks2p1);


    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2, x0_3, x1_3) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, x0_3, x1_3, rotations1[0]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2, x0_3, x1_3) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, x0_3, x1_3, rotations1[1]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2, x0_3, x1_3) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, x0_3, x1_3, rotations1[2]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2, x0_3, x1_3) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, x0_3, x1_3, rotations1[3]);

    const __m256i two = _mm256_set1_epi32(2u);
    const __m256i ks0p2 = _mm256_add_epi32(ks0, two);

    x0_0 = _mm256_add_epi32(x0_0,ks2);
    x0_1 = _mm256_add_epi32(x0_1,ks2);
    x0_2 = _mm256_add_epi32(x0_2,ks2);
    x0_3 = _mm256_add_epi32(x0_3,ks2);

    x1_0 = _mm256_add_epi32(x1_0, ks0p2);
    x1_1 = _mm256_add_epi32(x1_1, ks0p2);
    x1_2 = _mm256_add_epi32(x1_2, ks0p2);
    x1_3 = _mm256_add_epi32(x1_3, ks0p2);

    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2, x0_3, x1_3) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, x0_3, x1_3, rotations0[0]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2, x0_3, x1_3) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, x0_3, x1_3, rotations0[1]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2, x0_3, x1_3) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, x0_3, x1_3, rotations0[2]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2, x0_3, x1_3) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, x0_3, x1_3, rotations0[3]);

    const __m256i three = _mm256_set1_epi32(3u);
    const __m256i ks1p3 = _mm256_add_epi32(ks1, three);

    x0_0 = _mm256_add_epi32(x0_0,ks0);
    x0_1 = _mm256_add_epi32(x0_1,ks0);
    x0_2 = _mm256_add_epi32(x0_2,ks0);
    x0_3 = _mm256_add_epi32(x0_3,ks0);

    x1_0 = _mm256_add_epi32(x1_0, ks1p3);
    x1_1 = _mm256_add_epi32(x1_1, ks1p3);
    x1_2 = _mm256_add_epi32(x1_2, ks1p3);
    x1_3 = _mm256_add_epi32(x1_3, ks1p3);

    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2, x0_3, x1_3) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, x0_3, x1_3, rotations1[0]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2, x0_3, x1_3) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, x0_3, x1_3, rotations1[1]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2, x0_3, x1_3) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, x0_3, x1_3, rotations1[2]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2, x0_3, x1_3) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, x0_3, x1_3, rotations1[3]);

    const __m256i four = _mm256_set1_epi32(4u);
    const __m256i ks2p4 = _mm256_add_epi32(ks2, four);
    x0_0 = _mm256_add_epi32(x0_0,ks1);
    x0_1 = _mm256_add_epi32(x0_1,ks1);
    x0_2 = _mm256_add_epi32(x0_2,ks1);
    x0_3 = _mm256_add_epi32(x0_3,ks1);

    x1_0 = _mm256_add_epi32(x1_0,ks2p4);
    x1_1 = _mm256_add_epi32(x1_1,ks2p4);
    x1_2 = _mm256_add_epi32(x1_2,ks2p4);
    x1_3 = _mm256_add_epi32(x1_3,ks2p4);

    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2, x0_3, x1_3) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, x0_3, x1_3, rotations0[0]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2, x0_3, x1_3) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, x0_3, x1_3, rotations0[1]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2, x0_3, x1_3) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, x0_3, x1_3, rotations0[2]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2, x0_3, x1_3) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, x0_3, x1_3, rotations0[3]);

    const __m256i five = _mm256_set1_epi32(5u);
    const __m256i ks0p5 = _mm256_add_epi32(ks0, five);

    __m256i y0_0 = _mm256_add_epi32(x0_0,ks2);
    __m256i y0_1 = _mm256_add_epi32(x0_1,ks2);
    __m256i y0_2 = _mm256_add_epi32(x0_2,ks2);
    __m256i y0_3 = _mm256_add_epi32(x0_3,ks2);

    __m256i y1_0 =  _mm256_add_epi32(x1_0, ks0p5);
    __m256i y1_1 =  _mm256_add_epi32(x1_1, ks0p5);
    __m256i y1_2 =  _mm256_add_epi32(x1_2, ks0p5);
    __m256i y1_3 =  _mm256_add_epi32(x1_3, ks0p5);

    _mm256_storeu_si256((__m256i*) &out0[idx], y0_0);
    _mm256_storeu_si256((__m256i*) &out0[idx+8], y0_1);
    _mm256_storeu_si256((__m256i*) &out0[idx+16], y0_2);
    _mm256_storeu_si256((__m256i*) &out0[idx+24], y0_3);

    _mm256_storeu_si256((__m256i*) &out1[idx], y1_0);
    _mm256_storeu_si256((__m256i*) &out1[idx+8], y1_1);
    _mm256_storeu_si256((__m256i*) &out1[idx+16], y1_2);
    _mm256_storeu_si256((__m256i*) &out1[idx+24], y1_3);

  }
}


void ThreeFry2x32Kernel3(const std::uint32_t* key0,
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

  for (std::int64_t idx = 0; idx < n; idx += 24) {

    __m256i x0_0 = _mm256_loadu_si256((__m256i*) &data0[idx]);
    __m256i x0_1 = _mm256_loadu_si256((__m256i*) &data0[idx+8]);
    __m256i x0_2 = _mm256_loadu_si256((__m256i*) &data0[idx+16]);

    __m256i x1_0 = _mm256_loadu_si256((__m256i*) &data1[idx]);
    __m256i x1_1 = _mm256_loadu_si256((__m256i*) &data1[idx+8]);
    __m256i x1_2 = _mm256_loadu_si256((__m256i*) &data1[idx+16]);

    auto round = [](const __m256i v0_0, const __m256i v1_0, const __m256i v0_1, const __m256i v1_1, const __m256i v0_2, const __m256i v1_2, const std::uint8_t rotation) {
      const __m256i v0a_0 = _mm256_add_epi32(v0_0, v1_0); // lat 1, tp 3
      const __m256i v0a_1 = _mm256_add_epi32(v0_1, v1_1); // lat 1, tp 3
      const __m256i v0a_2 = _mm256_add_epi32(v0_2, v1_2); // lat 1, tp 3

      const __m256i w1_0 = _mm256_slli_epi32(v1_0, rotation); // lat 1, tp 2
      const __m256i w1_1 = _mm256_slli_epi32(v1_1, rotation); // lat 1, tp 2
      const __m256i w1_2 = _mm256_slli_epi32(v1_2, rotation); // lat 1, tp 2

      const __m256i w2_0 = _mm256_srli_epi32(v1_0, 32-rotation);
      const __m256i w2_1 = _mm256_srli_epi32(v1_1, 32-rotation);
      const __m256i w2_2 = _mm256_srli_epi32(v1_2, 32-rotation);

      const __m256i v1a_0 = _mm256_or_si256(w1_0, w2_0); // lat 1, tp 3
      const __m256i v1a_1 = _mm256_or_si256(w1_1, w2_1); // lat 1, tp 3
      const __m256i v1a_2 = _mm256_or_si256(w1_2, w2_2); // lat 1, tp 3

      const __m256i v1b_0 = _mm256_xor_si256(v1a_0, v0a_0); // lat 1, tp 3
      const __m256i v1b_1 = _mm256_xor_si256(v1a_1, v0a_1); // lat 1, tp 3
      const __m256i v1b_2 = _mm256_xor_si256(v1a_2, v0a_2); // lat 1, tp 3

      return std::make_tuple(v0a_0, v1b_0, v0a_1, v1b_1,v0a_2, v1b_2);
    };

    x0_0 = _mm256_add_epi32(x0_0,ks0); // lat 1, tp 3
    x0_1 = _mm256_add_epi32(x0_1,ks0); // lat 1, tp 3
    x0_2 = _mm256_add_epi32(x0_2,ks0); // lat 1, tp 3

    x1_0 = _mm256_add_epi32(x1_0,ks1); // lat 1, tp 3
    x1_1 = _mm256_add_epi32(x1_1,ks1); // lat 1, tp 3
    x1_2 = _mm256_add_epi32(x1_2,ks1); // lat 1, tp 3

    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, rotations0[0]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, rotations0[1]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, rotations0[2]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, rotations0[3]);

    const __m256i one = _mm256_set1_epi32(1u);
    const __m256i ks2p1 = _mm256_add_epi32(ks2, one);

    x0_0 = _mm256_add_epi32(x0_0,ks1);
    x0_1 = _mm256_add_epi32(x0_1,ks1);
    x0_2 = _mm256_add_epi32(x0_2,ks1);

    x1_0 = _mm256_add_epi32(x1_0,ks2p1);
    x1_1 = _mm256_add_epi32(x1_1,ks2p1);
    x1_2 = _mm256_add_epi32(x1_2,ks2p1);


    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, rotations1[0]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, rotations1[1]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, rotations1[2]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, rotations1[3]);

    const __m256i two = _mm256_set1_epi32(2u);
    const __m256i ks0p2 = _mm256_add_epi32(ks0, two);

    x0_0 = _mm256_add_epi32(x0_0,ks2);
    x0_1 = _mm256_add_epi32(x0_1,ks2);
    x0_2 = _mm256_add_epi32(x0_2,ks2);

    x1_0 = _mm256_add_epi32(x1_0, ks0p2);
    x1_1 = _mm256_add_epi32(x1_1, ks0p2);
    x1_2 = _mm256_add_epi32(x1_2, ks0p2);

    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, rotations0[0]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, rotations0[1]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, rotations0[2]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, rotations0[3]);

    const __m256i three = _mm256_set1_epi32(3u);
    const __m256i ks1p3 = _mm256_add_epi32(ks1, three);

    x0_0 = _mm256_add_epi32(x0_0,ks0);
    x0_1 = _mm256_add_epi32(x0_1,ks0);
    x0_2 = _mm256_add_epi32(x0_2,ks0);

    x1_0 = _mm256_add_epi32(x1_0, ks1p3);
    x1_1 = _mm256_add_epi32(x1_1, ks1p3);
    x1_2 = _mm256_add_epi32(x1_2, ks1p3);

    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, rotations1[0]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, rotations1[1]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, rotations1[2]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, rotations1[3]);

    const __m256i four = _mm256_set1_epi32(4u);
    const __m256i ks2p4 = _mm256_add_epi32(ks2, four);
    x0_0 = _mm256_add_epi32(x0_0,ks1);
    x0_1 = _mm256_add_epi32(x0_1,ks1);
    x0_2 = _mm256_add_epi32(x0_2,ks1);

    x1_0 = _mm256_add_epi32(x1_0,ks2p4);
    x1_1 = _mm256_add_epi32(x1_1,ks2p4);
    x1_2 = _mm256_add_epi32(x1_2,ks2p4);

    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, rotations0[0]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, rotations0[1]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, rotations0[2]);
    std::tie(x0_0, x1_0, x0_1, x1_1, x0_2, x1_2) = round(x0_0, x1_0, x0_1, x1_1,x0_2, x1_2, rotations0[3]);

    const __m256i five = _mm256_set1_epi32(5u);
    const __m256i ks0p5 = _mm256_add_epi32(ks0, five);

    __m256i y0_0 = _mm256_add_epi32(x0_0,ks2);
    __m256i y0_1 = _mm256_add_epi32(x0_1,ks2);
    __m256i y0_2 = _mm256_add_epi32(x0_2,ks2);

    __m256i y1_0 =  _mm256_add_epi32(x1_0, ks0p5);
    __m256i y1_1 =  _mm256_add_epi32(x1_1, ks0p5);
    __m256i y1_2 =  _mm256_add_epi32(x1_2, ks0p5);

    _mm256_storeu_si256((__m256i*) &out0[idx], y0_0);
    _mm256_storeu_si256((__m256i*) &out0[idx+8], y0_1);
    _mm256_storeu_si256((__m256i*) &out0[idx+16], y0_2);

    _mm256_storeu_si256((__m256i*) &out1[idx], y1_0);
    _mm256_storeu_si256((__m256i*) &out1[idx+8], y1_1);
    _mm256_storeu_si256((__m256i*) &out1[idx+16], y1_2);

  }
}







void ThreeFry2x32Kernel2(const std::uint32_t* key0,
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

  for (std::int64_t idx = 0; idx < n; idx += 16) {

    __m256i x0_0 = _mm256_loadu_si256((__m256i*) &data0[idx]);
    __m256i x0_1 = _mm256_loadu_si256((__m256i*) &data0[idx+8]);
    __m256i x1_0 = _mm256_loadu_si256((__m256i*) &data1[idx]);
    __m256i x1_1 = _mm256_loadu_si256((__m256i*) &data1[idx+8]);

    auto round = [](const __m256i v0_0, const __m256i v1_0, const __m256i v0_1, const __m256i v1_1, const std::uint8_t rotation) {
      const __m256i v0a_0 = _mm256_add_epi32(v0_0, v1_0); // lat 1, tp 3
      const __m256i v0a_1 = _mm256_add_epi32(v0_1, v1_1); // lat 1, tp 3

      const __m256i w1_0 = _mm256_slli_epi32(v1_0, rotation); // lat 1, tp 2
      const __m256i w1_1 = _mm256_slli_epi32(v1_1, rotation); // lat 1, tp 2

      const __m256i w2_0 = _mm256_srli_epi32(v1_0, 32-rotation);
      const __m256i w2_1 = _mm256_srli_epi32(v1_1, 32-rotation);

      const __m256i v1a_0 = _mm256_or_si256(w1_0, w2_0); // lat 1, tp 3
      const __m256i v1a_1 = _mm256_or_si256(w1_1, w2_1); // lat 1, tp 3

      const __m256i v1b_0 = _mm256_xor_si256(v1a_0, v0a_0); // lat 1, tp 3
      const __m256i v1b_1 = _mm256_xor_si256(v1a_1, v0a_1); // lat 1, tp 3

      return std::make_tuple(v0a_0, v1b_0, v0a_1, v1b_1);
    };

    x0_0 = _mm256_add_epi32(x0_0,ks0); // lat 1, tp 3
    x0_1 = _mm256_add_epi32(x0_1,ks0); // lat 1, tp 3
    x1_0 = _mm256_add_epi32(x1_0,ks1); // lat 1, tp 3
    x1_1 = _mm256_add_epi32(x1_1,ks1); // lat 1, tp 3

    std::tie(x0_0, x1_0, x0_1, x1_1) = round(x0_0, x1_0, x0_1, x1_1, rotations0[0]);
    std::tie(x0_0, x1_0, x0_1, x1_1) = round(x0_0, x1_0, x0_1, x1_1, rotations0[1]);
    std::tie(x0_0, x1_0, x0_1, x1_1) = round(x0_0, x1_0, x0_1, x1_1, rotations0[2]);
    std::tie(x0_0, x1_0, x0_1, x1_1) = round(x0_0, x1_0, x0_1, x1_1, rotations0[3]);

    const __m256i one = _mm256_set1_epi32(1u);
    const __m256i ks2p1 = _mm256_add_epi32(ks2, one);
    x0_0 = _mm256_add_epi32(x0_0,ks1);
    x0_1 = _mm256_add_epi32(x0_1,ks1);
    x1_0 = _mm256_add_epi32(x1_0,ks2p1);
    x1_1 = _mm256_add_epi32(x1_1,ks2p1);

    std::tie(x0_0, x1_0, x0_1, x1_1) = round(x0_0, x1_0, x0_1, x1_1, rotations1[0]);
    std::tie(x0_0, x1_0, x0_1, x1_1) = round(x0_0, x1_0, x0_1, x1_1, rotations1[1]);
    std::tie(x0_0, x1_0, x0_1, x1_1) = round(x0_0, x1_0, x0_1, x1_1, rotations1[2]);
    std::tie(x0_0, x1_0, x0_1, x1_1) = round(x0_0, x1_0, x0_1, x1_1, rotations1[3]);

    const __m256i two = _mm256_set1_epi32(2u);
    const __m256i ks0p2 = _mm256_add_epi32(ks0, two);
    x0_0 = _mm256_add_epi32(x0_0,ks2);
    x0_1 = _mm256_add_epi32(x0_1,ks2);
    x1_0 = _mm256_add_epi32(x1_0, ks0p2);
    x1_1 = _mm256_add_epi32(x1_1, ks0p2);

    std::tie(x0_0, x1_0, x0_1, x1_1) = round(x0_0, x1_0, x0_1, x1_1, rotations0[0]);
    std::tie(x0_0, x1_0, x0_1, x1_1) = round(x0_0, x1_0, x0_1, x1_1, rotations0[1]);
    std::tie(x0_0, x1_0, x0_1, x1_1) = round(x0_0, x1_0, x0_1, x1_1, rotations0[2]);
    std::tie(x0_0, x1_0, x0_1, x1_1) = round(x0_0, x1_0, x0_1, x1_1, rotations0[3]);

    const __m256i three = _mm256_set1_epi32(3u);
    const __m256i ks1p3 = _mm256_add_epi32(ks1, three);
    x0_0 = _mm256_add_epi32(x0_0,ks0);
    x0_1 = _mm256_add_epi32(x0_1,ks0);
    x1_0 = _mm256_add_epi32(x1_0, ks1p3);
    x1_1 = _mm256_add_epi32(x1_1, ks1p3);

    std::tie(x0_0, x1_0, x0_1, x1_1) = round(x0_0, x1_0, x0_1, x1_1, rotations1[0]);
    std::tie(x0_0, x1_0, x0_1, x1_1) = round(x0_0, x1_0, x0_1, x1_1, rotations1[1]);
    std::tie(x0_0, x1_0, x0_1, x1_1) = round(x0_0, x1_0, x0_1, x1_1, rotations1[2]);
    std::tie(x0_0, x1_0, x0_1, x1_1) = round(x0_0, x1_0, x0_1, x1_1, rotations1[3]);

    const __m256i four = _mm256_set1_epi32(4u);
    const __m256i ks2p4 = _mm256_add_epi32(ks2, four);
    x0_0 = _mm256_add_epi32(x0_0,ks1);
    x0_1 = _mm256_add_epi32(x0_1,ks1);
    x1_0 = _mm256_add_epi32(x1_0,ks2p4);
    x1_1 = _mm256_add_epi32(x1_1,ks2p4);

    std::tie(x0_0, x1_0, x0_1, x1_1) = round(x0_0, x1_0, x0_1, x1_1, rotations0[0]);
    std::tie(x0_0, x1_0, x0_1, x1_1) = round(x0_0, x1_0, x0_1, x1_1, rotations0[1]);
    std::tie(x0_0, x1_0, x0_1, x1_1) = round(x0_0, x1_0, x0_1, x1_1, rotations0[2]);
    std::tie(x0_0, x1_0, x0_1, x1_1) = round(x0_0, x1_0, x0_1, x1_1, rotations0[3]);

    const __m256i five = _mm256_set1_epi32(5u);
    const __m256i ks0p5 = _mm256_add_epi32(ks0, five);
    __m256i y0_0 = _mm256_add_epi32(x0_0,ks2);
    __m256i y0_1 = _mm256_add_epi32(x0_1,ks2);
    __m256i y1_0 =  _mm256_add_epi32(x1_0, ks0p5);
    __m256i y1_1 =  _mm256_add_epi32(x1_1, ks0p5);
    _mm256_storeu_si256((__m256i*) &out0[idx], y0_0);
    _mm256_storeu_si256((__m256i*) &out0[idx+8], y0_1);
    _mm256_storeu_si256((__m256i*) &out1[idx+8], y1_1);
    _mm256_storeu_si256((__m256i*) &out1[idx], y1_0);

  }
}


void ThreeFry2x32Kernel1(const std::uint32_t* key0,
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

      // TODO some of these can be done with just one call to PSHUFD / _mm256_shuffle_epi8
      // (e.g. distance=16)
      // whiich is port 5 only however
      // clang uses:
      // b = [2,3,0,1,  6,7,4,5,  10,11,8,9,  14,15,12,13,  18,19,16,17,  22,23,20,21,  26,27,24,25,  30,31,28,29]
      // b = [1,2,3,0,  5,6,7,4,  9,10,11,8,  13,14,15,12,  17,18,19,16,  21,22,23,20,  25,26,27,24,  29,30,31,28]

      const __m256i v1 = _mm256_slli_epi32(v, distance); // lat 1, tp 2
      const __m256i v2 = _mm256_srli_epi32(v, 32-distance);
      const __m256i v3 = _mm256_or_si256(v1, v2); // lat 1, tp 3
      return v3;
    };

    auto round = [&rotate_left](__m256i v0, __m256i v1, const std::uint8_t rotation) {

      const __m256i v0a =  _mm256_add_epi32(v0, v1); // lat 1, tp 3
      const __m256i v1a = rotate_left(v1, rotation);
      const __m256i v1b =  _mm256_xor_si256(v1a, v0a); // lat 1, tp 3
      return std::make_pair(v0a, v1b);
    };

    x0 = _mm256_add_epi32(x0,ks0); // lat 1, tp 3
    x1 = _mm256_add_epi32(x1,ks1); // lat 1, tp 3

    std::tie(x0, x1) = round(x0, x1, rotations0[0]);
    std::tie(x0, x1) = round(x0, x1, rotations0[1]);
    std::tie(x0, x1) = round(x0, x1, rotations0[2]);
    std::tie(x0, x1) = round(x0, x1, rotations0[3]);

    const __m256i one = _mm256_set1_epi32(1u);
    const __m256i ks2p1 = _mm256_add_epi32(ks2, one);
    x0 = _mm256_add_epi32(x0,ks1);
    x1 = _mm256_add_epi32(x1,ks2p1);

    std::tie(x0, x1) = round(x0, x1, rotations1[0]);
    std::tie(x0, x1) = round(x0, x1, rotations1[1]);
    std::tie(x0, x1) = round(x0, x1, rotations1[2]);
    std::tie(x0, x1) = round(x0, x1, rotations1[3]);

    const __m256i two = _mm256_set1_epi32(2u);
    const __m256i ks0p2 = _mm256_add_epi32(ks0, two);
    x0 = _mm256_add_epi32(x0,ks2);
    x1 = _mm256_add_epi32(x1, ks0p2);

    std::tie(x0, x1) = round(x0, x1, rotations0[0]);
    std::tie(x0, x1) = round(x0, x1, rotations0[1]);
    std::tie(x0, x1) = round(x0, x1, rotations0[2]);
    std::tie(x0, x1) = round(x0, x1, rotations0[3]);

    const __m256i three = _mm256_set1_epi32(3u);
    const __m256i ks1p3 = _mm256_add_epi32(ks1, three);
    x0 = _mm256_add_epi32(x0,ks0);
    x1 = _mm256_add_epi32(x1, ks1p3);

    std::tie(x0, x1) = round(x0, x1, rotations1[0]);
    std::tie(x0, x1) = round(x0, x1, rotations1[1]);
    std::tie(x0, x1) = round(x0, x1, rotations1[2]);
    std::tie(x0, x1) = round(x0, x1, rotations1[3]);


    const __m256i four = _mm256_set1_epi32(4u);
    const __m256i ks2p4 = _mm256_add_epi32(ks2, four);
    x0 = _mm256_add_epi32(x0,ks1);
    x1 = _mm256_add_epi32(x1,ks2p4);

    std::tie(x0, x1) = round(x0, x1, rotations0[0]);
    std::tie(x0, x1) = round(x0, x1, rotations0[1]);
    std::tie(x0, x1) = round(x0, x1, rotations0[2]);
    std::tie(x0, x1) = round(x0, x1, rotations0[3]);


    const __m256i five = _mm256_set1_epi32(5u);
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
