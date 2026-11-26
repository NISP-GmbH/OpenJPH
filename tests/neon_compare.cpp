//***************************************************************************/
// This software is released under the 2-Clause BSD license, included
// below.
//
// Copyright (c) 2026, NISP GmbH
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//***************************************************************************/
// File: neon_compare.cpp
//
// Compares every NEON kernel with the generic C function it replaces, on
// random input: the outputs must be equal bit for bit. AArch64 only; build
// against the static library and its sources, e.g.
//
//   c++ -O2 -std=c++14 -Isrc/core/openjph \
//       tests/neon_compare.cpp build/src/core/libopenjph.a -o neon_compare
//   ./neon_compare [iterations]
//
// The library must be built with the NEON kernels (OJPH_DISABLE_NEON off).
//***************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#include "ojph_arch.h"
#include "ojph_mem.h"
#include "ojph_version.h"

namespace ojph {
  namespace local {
    void ojph_encode_codeblock32(ui32* buf, ui32 missing_msbs,
      ui32 num_passes, ui32 width, ui32 height, ui32 stride, ui32* lengths,
      ojph::mem_elastic_allocator *elastic, ojph::coded_lists *& coded);
    void ojph_encode_codeblock_neon(ui32* buf, ui32 missing_msbs,
      ui32 num_passes, ui32 width, ui32 height, ui32 stride, ui32* lengths,
      ojph::mem_elastic_allocator *elastic, ojph::coded_lists *& coded);
    bool initialize_block_encoder_tables();
    bool initialize_block_encoder_tables_neon();
    bool ojph_decode_codeblock32(ui8* coded_data, ui32* decoded_data,
      ui32 missing_msbs, ui32 num_passes, ui32 lengths1, ui32 lengths2,
      ui32 width, ui32 height, ui32 stride, bool stripe_causal);
    bool ojph_decode_codeblock_neon(ui8* coded_data, ui32* decoded_data,
      ui32 missing_msbs, ui32 num_passes, ui32 lengths1, ui32 lengths2,
      ui32 width, ui32 height, ui32 stride, bool stripe_causal);
  }
}

using namespace ojph;
using namespace ojph::local;

static std::mt19937_64 rng(12345);
static ui32 rnd(ui32 n) { return (ui32)(rng() % n); }
static int failures = 0;

#define CHECK(cond, ...) do { if (!(cond)) { \
  if (failures++ < 20) { printf("FAIL %s:%d: ", __FILE__, __LINE__); \
    printf(__VA_ARGS__); printf("\n"); } } } while (0)

//////////////////////////////////////////////////////////////////////////////
// a code block of sign-magnitude samples as tx_to_cb32 leaves them: the
// magnitude's K_max bits at the top of bits 30..0, the sign in bit 31
static void fill_codeblock(std::vector<ui32>& buf, ui32 width, ui32 height,
                           ui32 stride, ui32 K_max)
{
  ui32 shift = 31 - K_max;
  ui32 max_mag = (K_max >= 31) ? 0x7FFFFFFFu : ((1u << K_max) - 1);
  int kind = (int)rnd(6);
  double scale = std::ldexp(1.0, (int)rnd(K_max + 1)) * (0.25 + (double)rnd(100) / 50.0);
  double density = (double)rnd(1000) / 1000.0;
  std::fill(buf.begin(), buf.end(), 0xDEADBEEFu);  // beyond the block: junk
  for (ui32 y = 0; y < height; ++y)
    for (ui32 x = 0; x < width; ++x)
    {
      ui32 mag = 0;
      switch (kind) {
        case 0: mag = (ui32)(rng() & max_mag); break;         // dense
        case 1: mag = 0; break;                                // empty
        case 2: if ((double)rnd(1000) / 1000.0 < density)     // sparse
                  mag = (ui32)(rng() & max_mag);
                break;
        default: {                                             // laplacian
          double u = ((double)(rng() >> 11) + 0.5) / 9007199254740992.0;
          double m = -std::log(u) * scale;
          if ((double)rnd(1000) / 1000.0 > density) m *= 0.1;
          mag = m >= (double)max_mag ? max_mag : (ui32)m;
        }
      }
      ui32 sign = (rng() & 1) ? 0x80000000u : 0;
      buf[y * stride + x] = (mag ? sign : 0) | (mag << shift);
    }
  // at least one sample significant, as codeblock::encode requires
  ui32 x = rnd(width), y = rnd(height);
  if ((buf[y * stride + x] & 0x7FFFFFFFu) == 0)
    buf[y * stride + x] = (1u << shift) | ((rng() & 1) ? 0x80000000u : 0);
}

//////////////////////////////////////////////////////////////////////////////
// The generic encoder of 0.33.0 writes one byte past its e_val / cx_val
// arrays for code blocks wider than 1020 samples (fixed after 0.33.0, #433),
// and its output is then not to be trusted; up to 0.33.0 the blocks
// compared are at most 1020 wide.  (The NEON encoder was compared with the
// fixed generic one at 1024 too: equal.)
#if OPENJPH_VERSION_MAJOR == 0 && OPENJPH_VERSION_MINOR <= 33
static const ui32 max_encoder_width = 1020;
#else
static const ui32 max_encoder_width = 1024;
#endif

static void test_block_encoder(int iterations)
{
  initialize_block_encoder_tables();
  initialize_block_encoder_tables_neon();
  mem_elastic_allocator el_a(1 << 20), el_b(1 << 20);
  static const ui32 widths[] = { 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 31,
    32, 33, 47, 48, 63, 64, 65, 96, 127, 128, 129, 255, 256, 512, 1024 };
  long blocks = 0, bytes = 0;
  for (int it = 0; it < iterations; ++it)
  {
    ui32 width, height;
    if (it % 4 == 0) {  // the usual shapes
      static const ui32 shapes[][2] = { {64, 64}, {128, 32}, {32, 32},
        {64, 32}, {32, 64}, {256, 16}, {1024, 4}, {16, 256}, {4, 1024} };
      int s = (int)rnd(9);
      width = shapes[s][0]; height = shapes[s][1];
      if (rnd(2)) { width = 1 + rnd(width); height = 1 + rnd(height); }
    } else {
      width = widths[rnd(sizeof(widths) / sizeof(widths[0]))];
      ui32 hmax = ojph_min(1024u, 4096u / width);
      height = 1 + rnd(hmax);
    }
    width = ojph_min(width, max_encoder_width);
    ui32 stride = width + rnd(20);
    ui32 K_max = 1 + rnd(30);
    std::vector<ui32> buf(stride * height + 64);
    fill_codeblock(buf, width, height, stride, K_max);

    ui32 len_a[2] = {0, 0}, len_b[2] = {0, 0};
    coded_lists *ca = NULL, *cb = NULL;
    el_a.restart(); el_b.restart();
    std::vector<ui32> copy = buf;
    ojph_encode_codeblock32(buf.data(), K_max - 1, 1, width, height, stride,
                            len_a, &el_a, ca);
    ojph_encode_codeblock_neon(buf.data(), K_max - 1, 1, width, height,
                               stride, len_b, &el_b, cb);
    CHECK(copy == buf, "encoder changed its input");
    CHECK(len_a[0] == len_b[0], "encoder length %u vs %u (%ux%u K_max %u)",
          len_a[0], len_b[0], width, height, K_max);
    if (len_a[0] == len_b[0])
      CHECK(memcmp(ca->buf, cb->buf, len_a[0]) == 0,
            "encoder bytes differ (%ux%u K_max %u)", width, height, K_max);
    ++blocks; bytes += len_a[0];
  }
  printf("block encoder: %ld blocks, %ld bytes compared\n", blocks, bytes);
}

//////////////////////////////////////////////////////////////////////////////
// encodes random code blocks with the generic encoder and decodes them with
// both decoders; for 2 and 3 passes, random bytes follow as the SigProp and
// MagRef segments (the decoders read them as any other bits)
static void test_block_decoder(int iterations)
{
  initialize_block_encoder_tables();
  mem_elastic_allocator el(1 << 20);
  static const ui32 shapes[][2] = { {64, 64}, {128, 32}, {32, 32}, {64, 32},
    {32, 64}, {256, 16}, {1024, 4}, {16, 256}, {4, 1024}, {8, 8}, {4, 4} };
  long blocks = 0, multi = 0, samples = 0, bad = 0, bad_ok = 0, bad_diff = 0;
  for (int it = 0; it < iterations; ++it)
  {
    int s = (int)rnd(sizeof(shapes) / sizeof(shapes[0]));
    ui32 width = shapes[s][0], height = shapes[s][1];
    if (rnd(2)) { width = 1 + rnd(width); height = 1 + rnd(height); }
    width = ojph_min(width, max_encoder_width);
    ui32 stride = ((width + 7) & ~7u) + 8 * rnd(3);
    ui32 K_max = 1 + rnd(29);
    std::vector<ui32> buf(stride * height + 64);
    fill_codeblock(buf, width, height, stride, K_max);

    ui32 len[2] = {0, 0};
    coded_lists *c = NULL;
    el.restart();
    ojph_encode_codeblock32(buf.data(), K_max - 1, 1, width, height, stride,
                            len, &el, c);

    ui32 num_passes = 1, len2 = 0;
    if (K_max >= 3 && rnd(3) == 0) {  // add SigProp (and MagRef) bytes
      num_passes = 2 + rnd(2);
      len2 = 1 + rnd(2 * width * height / 8 + 8);
    }
    const ui32 pre = 64, post = 64;
    std::vector<ui8> coded(pre + len[0] + len2 + post);
    for (size_t i = 0; i < coded.size(); ++i)
      coded[i] = (ui8)rng();
    memcpy(coded.data() + pre, c->buf, len[0]);
    for (ui32 i = 0; i < len2; ++i) {   // no 0xFF followed by > 0x8F
      ui8 b = (ui8)rng();
      if (i > 0 && coded[pre + len[0] + i - 1] == 0xFF) b &= 0x7F;
      coded[pre + len[0] + i] = b;
    }
    // a quarter of the blocks damaged: 1..3 bytes of the cleanup segment
    // changed (not its last two, which give the MEL / VLC length)
    bool damaged = len[0] > 2 && rnd(4) == 0;
    if (damaged)
      for (ui32 k = 1 + rnd(3); k > 0; --k)
        coded[pre + rnd(len[0] - 2)] ^= (ui8)(1 + rnd(255));
    std::vector<ui8> coded2 = coded;

    ui32 out_h = height + 4, out_stride = stride + 16;
    std::vector<ui32> out_a(out_stride * out_h, 0x5A5A5A5Au);
    std::vector<ui32> out_b(out_stride * out_h, 0x5A5A5A5Au);
    bool stripe_causal = rnd(4) == 0;
    bool ra = ojph_decode_codeblock32(coded.data() + pre, out_a.data(),
      K_max - 1, num_passes, len[0], len2, width, height, out_stride,
      stripe_causal);
    bool rb = ojph_decode_codeblock_neon(coded2.data() + pre, out_b.data(),
      K_max - 1, num_passes, len[0], len2, width, height, out_stride,
      stripe_causal);
    if (damaged)
      bad_diff += ra != rb;
    else
      CHECK(ra == rb, "decoder result %d vs %d (%ux%u K_max %u passes %u)",
            (int)ra, (int)rb, width, height, K_max, num_passes);
    if (num_passes == 1 && !damaged)
      CHECK(ra, "decoder failed on a valid block (%ux%u K_max %u)",
            width, height, K_max);
    if (ra && rb) {
      // compare the decoded block (what the codestream uses)
      bool same = true;
      for (ui32 y = 0; y < height && same; ++y)
        same = memcmp(out_a.data() + y * out_stride,
                      out_b.data() + y * out_stride, width * 4) == 0;
      // a damaged block may decode otherwise, or fail where the other
      // decoder does not: OpenJPH's SSSE3 and AVX2 decoders differ from the
      // generic one on the same damaged blocks as this one does; they run
      // for the sanitizers' sake
      if (damaged)
        bad_diff += !same;
      else
        CHECK(same, "decoded samples differ (%ux%u K_max %u passes %u sc %d)",
              width, height, K_max, num_passes, (int)stripe_causal);
      if (num_passes == 1 && !damaged && same) {  // the samples coded back
        ui32 shift = 31 - K_max;
        for (ui32 y = 0; y < height && same; ++y)
          for (ui32 x = 0; x < width && same; ++x) {
            ui32 v = buf[y * stride + x], d = out_a[y * out_stride + x];
            ui32 mv = (v & 0x7FFFFFFFu) >> shift, md = (d & 0x7FFFFFFFu) >> shift;
            same = mv == md && (mv == 0 || (v >> 31) == (d >> 31));
          }
        CHECK(same, "decoded samples are not the coded ones (%ux%u K_max %u)",
              width, height, K_max);
      }
    }
    ++blocks; multi += num_passes > 1; samples += width * height;
    bad += damaged; bad_ok += damaged && ra;
  }
  printf("block decoder: %ld blocks (%ld with SigProp/MagRef, %ld damaged of "
         "which %ld decoded; %ld decoded or failed otherwise than by the "
         "generic decoder), %ld samples\n", blocks, multi, bad, bad_ok, bad_diff,
         samples);
}

//////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
  int iterations = argc > 1 ? atoi(argv[1]) : 20000;
  test_block_encoder(iterations);
  test_block_decoder(iterations);
  printf("%s: %d failures\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
