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

#include <cassert>
#include <string>

#include "ojph_arch.h"
#include "ojph_mem.h"
#include "ojph_message.h"
#include "ojph_base.h"
#include "ojph_params.h"
// the tests build lifting kernels of their own
#define private public
#include "../src/core/codestream/ojph_params_local.h"
#undef private
#include "../src/core/transform/ojph_colour_local.h"
#include "../src/core/transform/ojph_transform_local.h"

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
    void gen_mem_clear(void* addr, size_t count);
    void neon_mem_clear(void* addr, size_t count);
    ui32 gen_find_max_val32(ui32* address);
    ui32 neon_find_max_val32(ui32* address);
    ui64 gen_find_max_val64(ui64* address);
    ui64 neon_find_max_val64(ui64* address);
    void gen_rev_tx_to_cb32(const void *sp, ui32 *dp, ui32 K_max,
                            float delta_inv, ui32 count, ui32* max_val);
    void neon_rev_tx_to_cb32(const void *sp, ui32 *dp, ui32 K_max,
                             float delta_inv, ui32 count, ui32* max_val);
    void gen_irv_tx_to_cb32(const void *sp, ui32 *dp, ui32 K_max,
                            float delta_inv, ui32 count, ui32* max_val);
    void neon_irv_tx_to_cb32(const void *sp, ui32 *dp, ui32 K_max,
                             float delta_inv, ui32 count, ui32* max_val);
    void gen_rev_tx_to_cb64(const void *sp, ui64 *dp, ui32 K_max,
                            float delta_inv, ui32 count, ui64* max_val);
    void neon_rev_tx_to_cb64(const void *sp, ui64 *dp, ui32 K_max,
                             float delta_inv, ui32 count, ui64* max_val);
    void gen_rev_tx_from_cb32(const ui32 *sp, void *dp, ui32 K_max,
                              float delta, ui32 count);
    void neon_rev_tx_from_cb32(const ui32 *sp, void *dp, ui32 K_max,
                               float delta, ui32 count);
    void gen_irv_tx_from_cb32(const ui32 *sp, void *dp, ui32 K_max,
                              float delta, ui32 count);
    void neon_irv_tx_from_cb32(const ui32 *sp, void *dp, ui32 K_max,
                               float delta, ui32 count);
    void gen_rev_tx_from_cb64(const ui64 *sp, void *dp, ui32 K_max,
                              float delta, ui32 count);
    void neon_rev_tx_from_cb64(const ui64 *sp, void *dp, ui32 K_max,
                               float delta, ui32 count);
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
  long blocks = 0, multi = 0, samples = 0;
  for (int it = 0; it < iterations; ++it)
  {
    int s = (int)rnd(sizeof(shapes) / sizeof(shapes[0]));
    ui32 width = shapes[s][0], height = shapes[s][1];
    if (rnd(2)) { width = 1 + rnd(width); height = 1 + rnd(height); }
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
    CHECK(ra == rb, "decoder result %d vs %d (%ux%u K_max %u passes %u)",
          (int)ra, (int)rb, width, height, K_max, num_passes);
    if (num_passes == 1)
      CHECK(ra, "decoder failed on a valid block (%ux%u K_max %u)",
            width, height, K_max);
    if (ra && rb) {
      // compare the decoded block (what the codestream uses)
      bool same = true;
      for (ui32 y = 0; y < height && same; ++y)
        same = memcmp(out_a.data() + y * out_stride,
                      out_b.data() + y * out_stride, width * 4) == 0;
      CHECK(same, "decoded samples differ (%ux%u K_max %u passes %u sc %d)",
            width, height, K_max, num_passes, (int)stripe_causal);
      if (num_passes == 1 && same) {  // the samples coded: magnitudes back
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
  }
  printf("block decoder: %ld blocks (%ld with SigProp/MagRef), %ld samples\n",
         blocks, multi, samples);
}

//////////////////////////////////////////////////////////////////////////////
// line kernels: codestream (sample conversion to and from code blocks),
// colour (conversion, RCT, ICT) and transform (5/3 and 9/7 lifting)
//////////////////////////////////////////////////////////////////////////////

// a line buffer with room on both sides, filled with a pattern
struct test_line {
  std::vector<ui8> store;
  line_buf lb;
  test_line(ui32 width, ui32 flags, ui8 fill = 0xA5)
  {
    ui32 bytes = (flags & line_buf::LFT_64BIT) ? 8 : 4;
    store.assign((width + 2 * 64) * bytes, fill);
    lb.size = width; lb.pre_size = 64; lb.flags = flags;
    lb.i32 = (si32*)(store.data() + 64 * bytes);
  }
  si32 *i32() { return lb.i32; }
  si64 *i64() { return lb.i64; }
  float *f32() { return lb.f32; }
};

static bool same_bytes(const void *a, const void *b, size_t n)
{ return memcmp(a, b, n) == 0; }

static float rnd_float(float range)
{
  int kind = (int)rnd(8);
  if (kind == 0) return 0.0f;
  if (kind == 1) return (rnd(2) ? 1.0f : -1.0f) * range * 4.0f;  // outside
  double u = (double)(rng() >> 11) / 9007199254740992.0;
  return (float)((u * 2.0 - 1.0) * range);
}

static si64 rnd_int(int bits)
{
  if (bits >= 63) return (si64)rng();
  si64 lim = (si64)1 << bits;
  return (si64)(rng() % (ui64)(2 * lim)) - lim;
}

static void test_codestream_kernels(int iterations)
{
  long n = 0;
  for (int it = 0; it < iterations; ++it)
  {
    ui32 count = 1 + rnd(it % 8 == 0 ? 1100 : 70);
    ui32 K_max = 1 + rnd(30);
    // rev_tx_to_cb32: samples of K_max bits (and a sign)
    {
      std::vector<si32> s(count + 16);
      for (auto &v : s) v = (si32)rnd_int((int)K_max - 1);
      std::vector<ui32> da(count + 16, 7), db(count + 16, 7);
      ui32 ma[8] = {0}, mb[8] = {0};
      gen_rev_tx_to_cb32(s.data(), da.data(), K_max, 1.0f, count, ma);
      neon_rev_tx_to_cb32(s.data(), db.data(), K_max, 1.0f, count, mb);
      CHECK(same_bytes(da.data(), db.data(), count * 4),
            "rev_tx_to_cb32 count %u K_max %u", count, K_max);
      CHECK(gen_find_max_val32(ma) == neon_find_max_val32(mb),
            "rev_tx_to_cb32 max_val count %u K_max %u", count, K_max);
    }
    // irv_tx_to_cb32: floats times delta_inv, truncated
    {
      std::vector<float> s(count + 16);
      float range = (float)(1u << rnd(16));
      for (auto &v : s) v = rnd_float(1.0f);
      float delta_inv = range * (0.5f + (float)rnd(1000) / 1000.0f);
      std::vector<ui32> da(count + 16, 7), db(count + 16, 7);
      ui32 ma[8] = {0}, mb[8] = {0};
      gen_irv_tx_to_cb32(s.data(), da.data(), K_max, delta_inv, count, ma);
      neon_irv_tx_to_cb32(s.data(), db.data(), K_max, delta_inv, count, mb);
      CHECK(same_bytes(da.data(), db.data(), count * 4),
            "irv_tx_to_cb32 count %u", count);
      CHECK(gen_find_max_val32(ma) == neon_find_max_val32(mb),
            "irv_tx_to_cb32 max_val count %u", count);
    }
    // rev_tx_from_cb32 / irv_tx_from_cb32: sign-magnitude back
    {
      std::vector<ui32> s(count + 16);
      ui32 shift = 31 - K_max;
      for (auto &v : s) {
        ui32 mag = (ui32)(rng() & ((1u << K_max) - 1)) << shift;
        v = mag | ((rng() & 1) ? 0x80000000u : 0);
        if (rnd(8) == 0) v = (ui32)rng();   // anything
      }
      std::vector<si32> da(count + 16, 7), db(count + 16, 7);
      gen_rev_tx_from_cb32(s.data(), da.data(), K_max, 1.0f, count);
      neon_rev_tx_from_cb32(s.data(), db.data(), K_max, 1.0f, count);
      CHECK(same_bytes(da.data(), db.data(), count * 4),
            "rev_tx_from_cb32 count %u K_max %u", count, K_max);
      float delta = (float)rnd(1000) / 1e6f + 1e-7f;
      gen_irv_tx_from_cb32(s.data(), da.data(), K_max, delta, count);
      neon_irv_tx_from_cb32(s.data(), db.data(), K_max, delta, count);
      CHECK(same_bytes(da.data(), db.data(), count * 4),
            "irv_tx_from_cb32 count %u", count);
    }
    // the 64-bit paths
    {
      ui32 K64 = 1 + rnd(62);
      std::vector<si64> s(count + 16);
      for (auto &v : s) v = rnd_int((int)K64 - 1);
      std::vector<ui64> da(count + 16, 7), db(count + 16, 7);
      ui64 ma[4] = {0}, mb[4] = {0};
      gen_rev_tx_to_cb64(s.data(), da.data(), K64, 1.0f, count, ma);
      neon_rev_tx_to_cb64(s.data(), db.data(), K64, 1.0f, count, mb);
      CHECK(same_bytes(da.data(), db.data(), count * 8),
            "rev_tx_to_cb64 count %u K_max %u", count, K64);
      CHECK(gen_find_max_val64(ma) == neon_find_max_val64(mb),
            "rev_tx_to_cb64 max_val");
      std::vector<si64> ra(count + 16, 7), rb(count + 16, 7);
      gen_rev_tx_from_cb64(da.data(), ra.data(), K64, 1.0f, count);
      neon_rev_tx_from_cb64(da.data(), rb.data(), K64, 1.0f, count);
      CHECK(same_bytes(ra.data(), rb.data(), count * 8),
            "rev_tx_from_cb64 count %u K_max %u", count, K64);
    }
    // mem_clear
    {
      std::vector<ui8> a(count * 4 + 64, 0x77), b(count * 4 + 64, 0x77);
      gen_mem_clear(a.data(), count * 4);
      neon_mem_clear(b.data(), count * 4);
      CHECK(same_bytes(a.data(), b.data(), count * 4), "mem_clear %u", count);
    }
    ++n;
  }
  printf("codestream kernels: %ld rounds\n", n);
}

static void test_colour_kernels(int iterations)
{
  const ui32 I32 = line_buf::LFT_32BIT | line_buf::LFT_INTEGER;
  const ui32 I64 = line_buf::LFT_64BIT | line_buf::LFT_INTEGER;
  const ui32 F32 = line_buf::LFT_32BIT;
  long n = 0;
  for (int it = 0; it < iterations; ++it)
  {
    ui32 width = 1 + rnd(it % 8 == 0 ? 2100 : 70);
    ui32 so = rnd(8), dof = rnd(8);
    // rev_convert, rev_convert_nlt_type3: 32 -> 32, 32 -> 64, 64 -> 32
    for (int kind = 0; kind < 3; ++kind)
      for (int nlt = 0; nlt < 2; ++nlt)
      {
        ui32 sf = kind == 2 ? I64 : I32, df = kind == 1 ? I64 : I32;
        test_line src(width + so, sf), da(width + dof, df), db(width + dof, df);
        for (ui32 i = 0; i < width + so; ++i)
          if (sf == I64) src.i64()[i] = rnd_int(40); else src.i32()[i] = (si32)rnd_int(28);
        si64 shift = rnd_int(20);
        if (nlt) {
          gen_rev_convert_nlt_type3(&src.lb, so, &da.lb, dof, shift, width);
          neon_rev_convert_nlt_type3(&src.lb, so, &db.lb, dof, shift, width);
        } else {
          gen_rev_convert(&src.lb, so, &da.lb, dof, shift, width);
          neon_rev_convert(&src.lb, so, &db.lb, dof, shift, width);
        }
        size_t bytes = (df == I64 ? 8 : 4);
        CHECK(same_bytes(da.store.data() + (64 + dof) * bytes,
                         db.store.data() + (64 + dof) * bytes, width * bytes),
              "rev_convert%s kind %d width %u", nlt ? "_nlt_type3" : "", kind, width);
      }
    // irv_convert_to_integer (and nlt_type3): floats in about [-0.5, 0.5)
    for (int nlt = 0; nlt < 2; ++nlt)
    {
      ui32 bit_depth = 1 + rnd(32);
      bool is_signed = rnd(2) != 0;
      test_line src(width, F32), da(width + dof, I32), db(width + dof, I32);
      for (ui32 i = 0; i < width; ++i) src.f32()[i] = rnd_float(0.5f);
      if (nlt) {
        gen_irv_convert_to_integer_nlt_type3(&src.lb, &da.lb, dof, bit_depth, is_signed, width);
        neon_irv_convert_to_integer_nlt_type3(&src.lb, &db.lb, dof, bit_depth, is_signed, width);
      } else {
        gen_irv_convert_to_integer(&src.lb, &da.lb, dof, bit_depth, is_signed, width);
        neon_irv_convert_to_integer(&src.lb, &db.lb, dof, bit_depth, is_signed, width);
      }
      CHECK(same_bytes(da.i32() + dof, db.i32() + dof, width * 4),
            "irv_convert_to_integer%s width %u bit_depth %u signed %d",
            nlt ? "_nlt_type3" : "", width, bit_depth, (int)is_signed);
    }
    // irv_convert_to_float (and nlt_type3)
    for (int nlt = 0; nlt < 2; ++nlt)
    {
      ui32 bit_depth = 1 + rnd(32);
      bool is_signed = rnd(2) != 0;
      test_line src(width + so, I32), da(width, F32), db(width, F32);
      for (ui32 i = 0; i < width + so; ++i) {
        si64 v = is_signed ? rnd_int((int)bit_depth - 1)
                           : (si64)(rng() % (1ull << bit_depth));
        src.i32()[i] = (si32)v;
      }
      if (nlt) {
        gen_irv_convert_to_float_nlt_type3(&src.lb, so, &da.lb, bit_depth, is_signed, width);
        neon_irv_convert_to_float_nlt_type3(&src.lb, so, &db.lb, bit_depth, is_signed, width);
      } else {
        gen_irv_convert_to_float(&src.lb, so, &da.lb, bit_depth, is_signed, width);
        neon_irv_convert_to_float(&src.lb, so, &db.lb, bit_depth, is_signed, width);
      }
      CHECK(same_bytes(da.f32(), db.f32(), width * 4),
            "irv_convert_to_float%s width %u bit_depth %u signed %d",
            nlt ? "_nlt_type3" : "", width, bit_depth, (int)is_signed);
    }
    // rct_forward / rct_backward: 32-bit, or 64-bit components of 32-bit
    for (int wide = 0; wide < 2; ++wide)
    {
      ui32 cf = wide ? I64 : I32;
      test_line r(width, I32), g(width, I32), b(width, I32);
      for (ui32 i = 0; i < width; ++i) {
        r.i32()[i] = (si32)rnd_int(wide ? 30 : 20);
        g.i32()[i] = (si32)rnd_int(wide ? 30 : 20);
        b.i32()[i] = (si32)rnd_int(wide ? 30 : 20);
      }
      test_line ya(width, cf), cba(width, cf), cra(width, cf);
      test_line yb(width, cf), cbb(width, cf), crb(width, cf);
      gen_rct_forward(&r.lb, &g.lb, &b.lb, &ya.lb, &cba.lb, &cra.lb, width);
      neon_rct_forward(&r.lb, &g.lb, &b.lb, &yb.lb, &cbb.lb, &crb.lb, width);
      size_t bytes = wide ? 8 : 4;
      CHECK(same_bytes(ya.i32(), yb.i32(), width * bytes) &&
            same_bytes(cba.i32(), cbb.i32(), width * bytes) &&
            same_bytes(cra.i32(), crb.i32(), width * bytes),
            "rct_forward width %u wide %d", width, wide);
      test_line ra(width, I32), ga(width, I32), ba(width, I32);
      test_line rb(width, I32), gb(width, I32), bb(width, I32);
      gen_rct_backward(&ya.lb, &cba.lb, &cra.lb, &ra.lb, &ga.lb, &ba.lb, width);
      neon_rct_backward(&ya.lb, &cba.lb, &cra.lb, &rb.lb, &gb.lb, &bb.lb, width);
      CHECK(same_bytes(ra.i32(), rb.i32(), width * 4) &&
            same_bytes(ga.i32(), gb.i32(), width * 4) &&
            same_bytes(ba.i32(), bb.i32(), width * 4),
            "rct_backward width %u wide %d", width, wide);
    }
    // ict_forward / ict_backward
    {
      std::vector<float> r(width + 8), g(width + 8), b(width + 8);
      for (ui32 i = 0; i < width; ++i) {
        r[i] = rnd_float(0.5f); g[i] = rnd_float(0.5f); b[i] = rnd_float(0.5f);
      }
      std::vector<float> ya(width + 8), cba(width + 8), cra(width + 8);
      std::vector<float> yb(width + 8), cbb(width + 8), crb(width + 8);
      gen_ict_forward(r.data(), g.data(), b.data(), ya.data(), cba.data(), cra.data(), width);
      neon_ict_forward(r.data(), g.data(), b.data(), yb.data(), cbb.data(), crb.data(), width);
      CHECK(same_bytes(ya.data(), yb.data(), width * 4) &&
            same_bytes(cba.data(), cbb.data(), width * 4) &&
            same_bytes(cra.data(), crb.data(), width * 4),
            "ict_forward width %u", width);
      std::vector<float> ra(width + 8), ga(width + 8), ba(width + 8);
      std::vector<float> rb(width + 8), gb(width + 8), bb(width + 8);
      gen_ict_backward(ya.data(), cba.data(), cra.data(), ra.data(), ga.data(), ba.data(), width);
      neon_ict_backward(ya.data(), cba.data(), cra.data(), rb.data(), gb.data(), bb.data(), width);
      CHECK(same_bytes(ra.data(), rb.data(), width * 4) &&
            same_bytes(ga.data(), gb.data(), width * 4) &&
            same_bytes(ba.data(), bb.data(), width * 4),
            "ict_backward width %u", width);
    }
    ++n;
  }
  printf("colour kernels: %ld rounds\n", n);
}

// a lifting kernel: the standard 5/3 or 9/7 one, or random steps
static void make_atk(param_atk &atk, bool reversible)
{
  if (reversible) {
    if (rnd(3)) { atk.init_rev53(); return; }
    atk.Natk = (ui8)(1 + rnd(4));
    for (ui32 s = 0; s < atk.Natk; ++s) {
      int a = (int)rnd(5);
      atk.d[s].rev.Aatk = (si16)(a == 0 ? 1 : a == 1 ? -1 : (int)rnd_int(6));
      atk.d[s].rev.Batk = (si16)rnd_int(6);
      atk.d[s].rev.Eatk = (ui8)rnd(5);
    }
    atk.Satk = 0x1000;  // reversible
  } else {
    if (rnd(3)) { atk.init_irv97(); return; }
    atk.Natk = (ui8)(1 + rnd(4));
    for (ui32 s = 0; s < atk.Natk; ++s)
      atk.d[s].irv.Aatk = rnd_float(2.0f);
    atk.Katk = 0.5f + (float)rnd(1000) / 1000.0f;
    atk.Satk = 0;
  }
}

static void test_transform_kernels(int iterations)
{
  const ui32 I32 = line_buf::LFT_32BIT | line_buf::LFT_INTEGER;
  const ui32 I64 = line_buf::LFT_64BIT | line_buf::LFT_INTEGER;
  const ui32 F32 = line_buf::LFT_32BIT;
  long n = 0;
  for (int it = 0; it < iterations; ++it)
  {
    ui32 width = 1 + rnd(it % 8 == 0 ? 2100 : 70);
    bool even = rnd(2) != 0;
    bool synthesis = rnd(2) != 0;
    // reversible: vertical step, horizontal analysis and synthesis
    for (int wide = 0; wide < 2; ++wide)
    {
      ui32 f = wide ? I64 : I32;
      param_atk atk;
      make_atk(atk, true);
      const lifting_step *s = atk.get_step(rnd(atk.get_num_steps()));
      test_line sig(width, f), oth(width, f), aa(width, f), ab(width, f);
      for (ui32 i = 0; i < width; ++i) {
        si64 v0 = rnd_int(wide ? 40 : 24), v1 = rnd_int(wide ? 40 : 24), v2 = rnd_int(wide ? 40 : 24);
        if (wide) { sig.i64()[i] = v0; oth.i64()[i] = v1; aa.i64()[i] = ab.i64()[i] = v2; }
        else { sig.i32()[i] = (si32)v0; oth.i32()[i] = (si32)v1; aa.i32()[i] = ab.i32()[i] = (si32)v2; }
      }
      gen_rev_vert_step(s, &sig.lb, &oth.lb, &aa.lb, width, synthesis);
      neon_rev_vert_step(s, &sig.lb, &oth.lb, &ab.lb, width, synthesis);
      size_t bytes = wide ? 8 : 4;
      CHECK(same_bytes(aa.i32(), ab.i32(), width * bytes),
            "rev_vert_step width %u wide %d A %d B %d E %d syn %d", width, wide,
            s->rev.Aatk, s->rev.Batk, s->rev.Eatk, (int)synthesis);

      test_line src(width, f), la(width, f), ha(width, f), lb(width, f), hb(width, f);
      for (ui32 i = 0; i < width; ++i) {
        si64 v = rnd_int(wide ? 40 : 22);
        if (wide) src.i64()[i] = v; else src.i32()[i] = (si32)v;
      }
      test_line src2 = src; src2.lb.i32 = (si32*)(src2.store.data() + 64 * bytes);
      gen_rev_horz_ana(&atk, &la.lb, &ha.lb, &src.lb, width, even);
      neon_rev_horz_ana(&atk, &lb.lb, &hb.lb, &src2.lb, width, even);
      ui32 lw = (width + (even ? 1 : 0)) >> 1, hw = (width + (even ? 0 : 1)) >> 1;
      CHECK(same_bytes(la.i32(), lb.i32(), lw * bytes) &&
            same_bytes(ha.i32(), hb.i32(), hw * bytes),
            "rev_horz_ana width %u wide %d even %d steps %u", width, wide,
            (int)even, atk.get_num_steps());

      test_line da(width, f), db(width, f);
      test_line la2 = la, ha2 = ha;
      la2.lb.i32 = (si32*)(la2.store.data() + 64 * bytes);
      ha2.lb.i32 = (si32*)(ha2.store.data() + 64 * bytes);
      gen_rev_horz_syn(&atk, &da.lb, &la.lb, &ha.lb, width, even);
      neon_rev_horz_syn(&atk, &db.lb, &la2.lb, &ha2.lb, width, even);
      CHECK(same_bytes(da.i32(), db.i32(), width * bytes),
            "rev_horz_syn width %u wide %d even %d steps %u", width, wide,
            (int)even, atk.get_num_steps());
    }
    // irreversible
    {
      param_atk atk;
      make_atk(atk, false);
      const lifting_step *s = atk.get_step(rnd(atk.get_num_steps()));
      test_line sig(width, F32), oth(width, F32), aa(width, F32), ab(width, F32);
      for (ui32 i = 0; i < width; ++i) {
        sig.f32()[i] = rnd_float(1.0f); oth.f32()[i] = rnd_float(1.0f);
        aa.f32()[i] = ab.f32()[i] = rnd_float(1.0f);
      }
      gen_irv_vert_step(s, &sig.lb, &oth.lb, &aa.lb, width, synthesis);
      neon_irv_vert_step(s, &sig.lb, &oth.lb, &ab.lb, width, synthesis);
      CHECK(same_bytes(aa.f32(), ab.f32(), width * 4),
            "irv_vert_step width %u syn %d", width, (int)synthesis);
      float K = atk.get_K();
      gen_irv_vert_times_K(K, &aa.lb, width);
      neon_irv_vert_times_K(K, &ab.lb, width);
      CHECK(same_bytes(aa.f32(), ab.f32(), width * 4),
            "irv_vert_times_K width %u", width);

      test_line src(width, F32), la(width, F32), ha(width, F32), lb(width, F32), hb(width, F32);
      for (ui32 i = 0; i < width; ++i) src.f32()[i] = rnd_float(1.0f);
      test_line src2 = src; src2.lb.i32 = (si32*)(src2.store.data() + 64 * 4);
      gen_irv_horz_ana(&atk, &la.lb, &ha.lb, &src.lb, width, even);
      neon_irv_horz_ana(&atk, &lb.lb, &hb.lb, &src2.lb, width, even);
      ui32 lw = (width + (even ? 1 : 0)) >> 1, hw = (width + (even ? 0 : 1)) >> 1;
      CHECK(same_bytes(la.f32(), lb.f32(), lw * 4) &&
            same_bytes(ha.f32(), hb.f32(), hw * 4),
            "irv_horz_ana width %u even %d steps %u", width, (int)even,
            atk.get_num_steps());

      test_line da(width, F32), db(width, F32);
      test_line la2 = la, ha2 = ha;
      la2.lb.i32 = (si32*)(la2.store.data() + 64 * 4);
      ha2.lb.i32 = (si32*)(ha2.store.data() + 64 * 4);
      gen_irv_horz_syn(&atk, &da.lb, &la.lb, &ha.lb, width, even);
      neon_irv_horz_syn(&atk, &db.lb, &la2.lb, &ha2.lb, width, even);
      CHECK(same_bytes(da.f32(), db.f32(), width * 4),
            "irv_horz_syn width %u even %d steps %u", width, (int)even,
            atk.get_num_steps());
    }
    ++n;
  }
  printf("transform kernels: %ld rounds\n", n);
}

//////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
  int iterations = argc > 1 ? atoi(argv[1]) : 20000;
  test_block_encoder(iterations);
  test_block_decoder(iterations);
  test_codestream_kernels(iterations);
  test_colour_kernels(iterations);
  test_transform_kernels(iterations);
  printf("%s: %d failures\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
