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
// This file is part of the OpenJPH software implementation.
// File: ojph_simd_neon.h
//***************************************************************************/
//
// 128-bit operations for the NEON (AArch64 Advanced SIMD) kernels.
//
// Several NEON kernels are the WebAssembly SIMD kernels (*_wasm.cpp) built
// on these operations instead of <wasm_simd128.h>: each nv_<op> has the
// semantics of the WebAssembly SIMD instruction wasm_<op>, including its
// shift amounts taken modulo the lane width and its swizzle giving 0 for
// an index of 16 or more (as NEON's TBL does), so a kernel computes what
// its WebAssembly original computes.  One type, nv128_t, holds every lane
// layout, as v128_t does.
//***************************************************************************/

#ifndef OJPH_SIMD_NEON_H
#define OJPH_SIMD_NEON_H

#include <arm_neon.h>
#include <cstdint>
#include <cstring>

namespace ojph {
  namespace local {

    typedef uint8x16_t nv128_t;

    // reinterpretations
    static inline uint32x4_t nv_u32(nv128_t a) { return vreinterpretq_u32_u8(a); }
    static inline int32x4_t  nv_s32(nv128_t a) { return vreinterpretq_s32_u8(a); }
    static inline uint16x8_t nv_u16(nv128_t a) { return vreinterpretq_u16_u8(a); }
    static inline int16x8_t  nv_s16(nv128_t a) { return vreinterpretq_s16_u8(a); }
    static inline int8x16_t  nv_s8(nv128_t a)  { return vreinterpretq_s8_u8(a); }
    static inline uint64x2_t nv_u64(nv128_t a) { return vreinterpretq_u64_u8(a); }
    static inline int64x2_t  nv_s64(nv128_t a) { return vreinterpretq_s64_u8(a); }
    static inline float32x4_t nv_f32(nv128_t a) { return vreinterpretq_f32_u8(a); }
    static inline nv128_t nv_v(uint32x4_t a) { return vreinterpretq_u8_u32(a); }
    static inline nv128_t nv_v(int32x4_t a)  { return vreinterpretq_u8_s32(a); }
    static inline nv128_t nv_v(uint16x8_t a) { return vreinterpretq_u8_u16(a); }
    static inline nv128_t nv_v(int16x8_t a)  { return vreinterpretq_u8_s16(a); }
    static inline nv128_t nv_v(int8x16_t a)  { return vreinterpretq_u8_s8(a); }
    static inline nv128_t nv_v(uint8x16_t a) { return a; }
    static inline nv128_t nv_v(uint64x2_t a) { return vreinterpretq_u8_u64(a); }
    static inline nv128_t nv_v(int64x2_t a)  { return vreinterpretq_u8_s64(a); }
    static inline nv128_t nv_v(float32x4_t a) { return vreinterpretq_u8_f32(a); }

    // memory
    static inline nv128_t nv_v128_load(const void *p)
    { return vld1q_u8((const uint8_t*)p); }
    static inline void nv_v128_store(void *p, nv128_t a)
    { vst1q_u8((uint8_t*)p, a); }
    static inline void nv_v128_store32_lane0(void *p, nv128_t a)
    { uint32_t t = vgetq_lane_u32(nv_u32(a), 0); memcpy(p, &t, 4); }

    // bitwise
    static inline nv128_t nv_v128_and(nv128_t a, nv128_t b)
    { return vandq_u8(a, b); }
    static inline nv128_t nv_v128_or(nv128_t a, nv128_t b)
    { return vorrq_u8(a, b); }
    static inline nv128_t nv_v128_xor(nv128_t a, nv128_t b)
    { return veorq_u8(a, b); }
    // a & ~b, as wasm_v128_andnot
    static inline nv128_t nv_v128_andnot(nv128_t a, nv128_t b)
    { return vbicq_u8(a, b); }

    // constants and splats
    static inline nv128_t nv_i64x2_const(int64_t a, int64_t b)
    { return nv_v(vcombine_s64(vcreate_s64((uint64_t)a),
                               vcreate_s64((uint64_t)b))); }
    static inline nv128_t nv_u64x2_const(uint64_t a, uint64_t b)
    { return nv_v(vcombine_u64(vcreate_u64(a), vcreate_u64(b))); }
    static inline nv128_t nv_i32x4_const(int32_t a, int32_t b,
                                         int32_t c, int32_t d)
    { const int32_t t[4] = { a, b, c, d }; return nv_v(vld1q_s32(t)); }
    static inline nv128_t nv_u32x4_const(uint32_t a, uint32_t b,
                                         uint32_t c, uint32_t d)
    { const uint32_t t[4] = { a, b, c, d }; return nv_v(vld1q_u32(t)); }
    static inline nv128_t nv_i32x4_make(int32_t a, int32_t b,
                                        int32_t c, int32_t d)
    { return nv_i32x4_const(a, b, c, d); }
    static inline nv128_t nv_i16x8_const(int16_t a0, int16_t a1, int16_t a2,
      int16_t a3, int16_t a4, int16_t a5, int16_t a6, int16_t a7)
    { const int16_t t[8] = { a0, a1, a2, a3, a4, a5, a6, a7 };
      return nv_v(vld1q_s16(t)); }
    static inline nv128_t nv_u16x8_const(uint16_t a0, uint16_t a1,
      uint16_t a2, uint16_t a3, uint16_t a4, uint16_t a5, uint16_t a6,
      uint16_t a7)
    { const uint16_t t[8] = { a0, a1, a2, a3, a4, a5, a6, a7 };
      return nv_v(vld1q_u16(t)); }
    static inline nv128_t nv_i8x16_const(int8_t a0, int8_t a1, int8_t a2,
      int8_t a3, int8_t a4, int8_t a5, int8_t a6, int8_t a7, int8_t a8,
      int8_t a9, int8_t a10, int8_t a11, int8_t a12, int8_t a13,
      int8_t a14, int8_t a15)
    { const int8_t t[16] = { a0, a1, a2, a3, a4, a5, a6, a7,
                             a8, a9, a10, a11, a12, a13, a14, a15 };
      return nv_v(vld1q_s8(t)); }
    static inline nv128_t nv_i8x16_splat(int8_t a)
    { return nv_v(vdupq_n_s8(a)); }
    static inline nv128_t nv_i16x8_splat(int16_t a)
    { return nv_v(vdupq_n_s16(a)); }
    static inline nv128_t nv_i32x4_splat(int32_t a)
    { return nv_v(vdupq_n_s32(a)); }
    static inline nv128_t nv_u32x4_splat(uint32_t a)
    { return nv_v(vdupq_n_u32(a)); }
    static inline nv128_t nv_i64x2_splat(int64_t a)
    { return nv_v(vdupq_n_s64(a)); }
    static inline nv128_t nv_f32x4_splat(float a)
    { return nv_v(vdupq_n_f32(a)); }

    // lanes
    static inline uint8_t nv_u8x16_extract_lane0(nv128_t a)
    { return vgetq_lane_u8(a, 0); }
    #define nv_u8x16_extract_lane(a, i) vgetq_lane_u8((a), (i))
    #define nv_u16x8_extract_lane(a, i) vgetq_lane_u16(nv_u16(a), (i))
    #define nv_u32x4_extract_lane(a, i) vgetq_lane_u32(nv_u32(a), (i))
    #define nv_i32x4_extract_lane(a, i) vgetq_lane_s32(nv_s32(a), (i))
    #define nv_i64x2_extract_lane(a, i) vgetq_lane_s64(nv_s64(a), (i))

    // 8-bit lanes
    static inline nv128_t nv_i8x16_add(nv128_t a, nv128_t b)
    { return vaddq_u8(a, b); }
    static inline nv128_t nv_i8x16_eq(nv128_t a, nv128_t b)
    { return vceqq_u8(a, b); }
    static inline nv128_t nv_i8x16_gt(nv128_t a, nv128_t b)
    { return vcgtq_s8(nv_s8(a), nv_s8(b)); }
    static inline nv128_t nv_i8x16_abs(nv128_t a)
    { return nv_v(vabsq_s8(nv_s8(a))); }
    static inline nv128_t nv_u8x16_min(nv128_t a, nv128_t b)
    { return vminq_u8(a, b); }
    // wasm_i8x16_swizzle: an index of 16 or more gives 0, as TBL does
    static inline nv128_t nv_i8x16_swizzle(nv128_t a, nv128_t idx)
    { return vqtbl1q_u8(a, idx); }
    // one bit per lane, lane i at bit i, from the lane's top bit
    static inline uint32_t nv_i8x16_bitmask(nv128_t a)
    {
      const uint8x16_t bits = vreinterpretq_u8_u64(vcombine_u64(
        vcreate_u64(0x8040201008040201ull), vcreate_u64(0x8040201008040201ull)));
      uint8x16_t m = vandq_u8(vreinterpretq_u8_s8(vshrq_n_s8(nv_s8(a), 7)),
                              bits);
      return (uint32_t)vaddv_u8(vget_low_u8(m))
           | ((uint32_t)vaddv_u8(vget_high_u8(m)) << 8);
    }
    // true if every lane is all ones (a comparison's result)
    static inline bool nv_all_ones(nv128_t a)
    { return vminvq_u32(nv_u32(a)) == 0xFFFFFFFFu; }

    // 16-bit lanes
    static inline nv128_t nv_i16x8_add(nv128_t a, nv128_t b)
    { return nv_v(vaddq_u16(nv_u16(a), nv_u16(b))); }
    static inline nv128_t nv_i16x8_sub(nv128_t a, nv128_t b)
    { return nv_v(vsubq_u16(nv_u16(a), nv_u16(b))); }
    static inline nv128_t nv_i16x8_mul(nv128_t a, nv128_t b)
    { return nv_v(vmulq_u16(nv_u16(a), nv_u16(b))); }
    static inline nv128_t nv_i16x8_eq(nv128_t a, nv128_t b)
    { return nv_v(vceqq_u16(nv_u16(a), nv_u16(b))); }
    static inline nv128_t nv_i16x8_max(nv128_t a, nv128_t b)
    { return nv_v(vmaxq_s16(nv_s16(a), nv_s16(b))); }
    static inline nv128_t nv_i16x8_shl(nv128_t a, uint32_t n)
    { return nv_v(vshlq_u16(nv_u16(a), vdupq_n_s16((int16_t)(n & 15)))); }
    static inline nv128_t nv_u16x8_shr(nv128_t a, uint32_t n)
    { return nv_v(vshlq_u16(nv_u16(a), vdupq_n_s16((int16_t)-(int)(n & 15)))); }

    // 32-bit lanes
    static inline nv128_t nv_i32x4_add(nv128_t a, nv128_t b)
    { return nv_v(vaddq_u32(nv_u32(a), nv_u32(b))); }
    static inline nv128_t nv_i32x4_sub(nv128_t a, nv128_t b)
    { return nv_v(vsubq_u32(nv_u32(a), nv_u32(b))); }
    static inline nv128_t nv_i32x4_mul(nv128_t a, nv128_t b)
    { return nv_v(vmulq_u32(nv_u32(a), nv_u32(b))); }
    static inline nv128_t nv_i32x4_eq(nv128_t a, nv128_t b)
    { return nv_v(vceqq_u32(nv_u32(a), nv_u32(b))); }
    static inline nv128_t nv_i32x4_gt(nv128_t a, nv128_t b)
    { return nv_v(vcgtq_s32(nv_s32(a), nv_s32(b))); }
    static inline nv128_t nv_i32x4_lt(nv128_t a, nv128_t b)
    { return nv_v(vcltq_s32(nv_s32(a), nv_s32(b))); }
    static inline nv128_t nv_i32x4_min(nv128_t a, nv128_t b)
    { return nv_v(vminq_s32(nv_s32(a), nv_s32(b))); }
    static inline nv128_t nv_i32x4_max(nv128_t a, nv128_t b)
    { return nv_v(vmaxq_s32(nv_s32(a), nv_s32(b))); }
    static inline nv128_t nv_i32x4_shl(nv128_t a, uint32_t n)
    { return nv_v(vshlq_u32(nv_u32(a), vdupq_n_s32((int32_t)(n & 31)))); }
    static inline nv128_t nv_u32x4_shr(nv128_t a, uint32_t n)
    { return nv_v(vshlq_u32(nv_u32(a), vdupq_n_s32(-(int32_t)(n & 31)))); }
    static inline nv128_t nv_i32x4_shr(nv128_t a, uint32_t n)
    { return nv_v(vshlq_s32(nv_s32(a), vdupq_n_s32(-(int32_t)(n & 31)))); }
    // wasm_i32x4_trunc_sat_f32x4: toward zero, saturated, NaN to 0
    static inline nv128_t nv_i32x4_trunc_sat_f32x4(nv128_t a)
    { return nv_v(vcvtq_s32_f32(nv_f32(a))); }

    // 64-bit lanes
    static inline nv128_t nv_i64x2_add(nv128_t a, nv128_t b)
    { return nv_v(vaddq_u64(nv_u64(a), nv_u64(b))); }
    static inline nv128_t nv_i64x2_sub(nv128_t a, nv128_t b)
    { return nv_v(vsubq_u64(nv_u64(a), nv_u64(b))); }
    static inline nv128_t nv_i64x2_mul(nv128_t a, nv128_t b)
    {
      uint64_t x0 = vgetq_lane_u64(nv_u64(a), 0) * vgetq_lane_u64(nv_u64(b), 0);
      uint64_t x1 = vgetq_lane_u64(nv_u64(a), 1) * vgetq_lane_u64(nv_u64(b), 1);
      return nv_u64x2_const(x0, x1);
    }
    static inline nv128_t nv_i64x2_lt(nv128_t a, nv128_t b)
    { return nv_v(vcltq_s64(nv_s64(a), nv_s64(b))); }
    static inline nv128_t nv_i64x2_shl(nv128_t a, uint32_t n)
    { return nv_v(vshlq_u64(nv_u64(a), vdupq_n_s64((int64_t)(n & 63)))); }
    static inline nv128_t nv_u64x2_shr(nv128_t a, uint32_t n)
    { return nv_v(vshlq_u64(nv_u64(a), vdupq_n_s64(-(int64_t)(n & 63)))); }
    static inline nv128_t nv_i64x2_shr(nv128_t a, uint32_t n)
    { return nv_v(vshlq_s64(nv_s64(a), vdupq_n_s64(-(int64_t)(n & 63)))); }
    static inline nv128_t nv_i64x2_extend_low_i32x4(nv128_t a)
    { return nv_v(vmovl_s32(vget_low_s32(nv_s32(a)))); }
    static inline nv128_t nv_i64x2_extend_high_i32x4(nv128_t a)
    { return nv_v(vmovl_high_s32(nv_s32(a))); }

    // 32-bit floats
    static inline nv128_t nv_f32x4_add(nv128_t a, nv128_t b)
    { return nv_v(vaddq_f32(nv_f32(a), nv_f32(b))); }
    static inline nv128_t nv_f32x4_sub(nv128_t a, nv128_t b)
    { return nv_v(vsubq_f32(nv_f32(a), nv_f32(b))); }
    static inline nv128_t nv_f32x4_mul(nv128_t a, nv128_t b)
    { return nv_v(vmulq_f32(nv_f32(a), nv_f32(b))); }
    // c + a * b, fused (one rounding), as fmla
    static inline nv128_t nv_f32x4_fma(nv128_t c, nv128_t a, nv128_t b)
    { return nv_v(vfmaq_f32(nv_f32(c), nv_f32(a), nv_f32(b))); }
    // c - a * b, fused (one rounding), as fmls
    static inline nv128_t nv_f32x4_fms(nv128_t c, nv128_t a, nv128_t b)
    { return nv_v(vfmsq_f32(nv_f32(c), nv_f32(a), nv_f32(b))); }
    static inline nv128_t nv_f32x4_ge(nv128_t a, nv128_t b)
    { return nv_v(vcgeq_f32(nv_f32(a), nv_f32(b))); }
    static inline nv128_t nv_f32x4_lt(nv128_t a, nv128_t b)
    { return nv_v(vcltq_f32(nv_f32(a), nv_f32(b))); }
    static inline nv128_t nv_f32x4_convert_i32x4(nv128_t a)
    { return nv_v(vcvtq_f32_s32(nv_s32(a))); }

    // shuffles with constant lane indices: lanes 0..N-1 from a, N..2N-1
    // from b, as wasm_*_shuffle
#if defined(__has_builtin)
  #if __has_builtin(__builtin_shufflevector)
    #define OJPH_NV_HAS_SHUFFLEVECTOR
  #endif
#endif
#ifdef OJPH_NV_HAS_SHUFFLEVECTOR
    #define nv_i8x16_shuffle(a, b, ...) \
      ((nv128_t)__builtin_shufflevector((uint8x16_t)(a), (uint8x16_t)(b), \
        __VA_ARGS__))
    #define nv_i16x8_shuffle(a, b, ...) \
      nv_v((uint16x8_t)__builtin_shufflevector(nv_u16(a), nv_u16(b), \
        __VA_ARGS__))
    #define nv_i32x4_shuffle(a, b, ...) \
      nv_v((uint32x4_t)__builtin_shufflevector(nv_u32(a), nv_u32(b), \
        __VA_ARGS__))
    #define nv_i64x2_shuffle(a, b, ...) \
      nv_v((uint64x2_t)__builtin_shufflevector(nv_u64(a), nv_u64(b), \
        __VA_ARGS__))
#else
    // through a two-register table lookup
    static inline nv128_t nv_shuffle_bytes(nv128_t a, nv128_t b,
                                           const uint8_t *idx)
    { uint8x16x2_t t; t.val[0] = a; t.val[1] = b;
      return vqtbl2q_u8(t, vld1q_u8(idx)); }
    template <int W, int... I>
    static inline nv128_t nv_shuffle(nv128_t a, nv128_t b)
    {
      const int lanes[] = { I... };
      uint8_t idx[16];
      for (int i = 0; i < 16; ++i)
        idx[i] = (uint8_t)(lanes[i / W] * W + i % W);
      return nv_shuffle_bytes(a, b, idx);
    }
    #define nv_i8x16_shuffle(a, b, ...) nv_shuffle<1, __VA_ARGS__>((a), (b))
    #define nv_i16x8_shuffle(a, b, ...) nv_shuffle<2, __VA_ARGS__>((a), (b))
    #define nv_i32x4_shuffle(a, b, ...) nv_shuffle<4, __VA_ARGS__>((a), (b))
    #define nv_i64x2_shuffle(a, b, ...) nv_shuffle<8, __VA_ARGS__>((a), (b))
#endif

  }
}

#endif // !OJPH_SIMD_NEON_H
