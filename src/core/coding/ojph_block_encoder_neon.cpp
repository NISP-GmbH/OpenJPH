//***************************************************************************/
// This software is released under the 2-Clause BSD license, included
// below.
//
// Copyright (c) 2019, Aous Naman
// Copyright (c) 2019, Kakadu Software Pty Ltd, Australia
// Copyright (c) 2019, The University of New South Wales, Australia
// Copyright (c) 2024, Intel Corporation
// Copyright (c) 2026, Osamu Watanabe
// Copyright (c) 2026, NISP GmbH (NEON port)
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
// File: ojph_block_encoder_neon.cpp
//***************************************************************************/

// A NEON (AArch64 Advanced SIMD) port of ojph_block_encoder_avx2.cpp.
// The scalar MEL, VLC and MagSgn writers are those of the AVX2 encoder;
// each 256-bit AVX2 vector (8 quads of a quad row) is held here as two
// 128-bit NEON vectors, lanes 0..3 in lo and 4..7 in hi, so the order in
// which codewords are emitted, and so the codestream, is the same as the
// generic encoder's (ojph_block_encoder.cpp).

#include "ojph_arch.h"
#if defined(OJPH_ENABLE_NEON)

#include <cassert>
#include <cstring>
#include <cstdint>
#include <climits>
#include <arm_neon.h>
#include <mutex>

#include "ojph_mem.h"
#include "ojph_arch.h"
#include "ojph_block_encoder.h"
#include "ojph_message.h"

#ifdef OJPH_COMPILER_MSVC
  #define likely(x)       (x)
  #define unlikely(x)     (x)
#else
  #define likely(x)       __builtin_expect((x), 1)
  #define unlikely(x)     __builtin_expect((x), 0)
#endif

namespace ojph {
  namespace local {

    /////////////////////////////////////////////////////////////////////////
    // tables
    /////////////////////////////////////////////////////////////////////////

    //VLC encoding
    // index is (c_q << 8) + (rho << 4) + eps
    // data is  (cwd << 8) + (cwd_len << 4) + eps
    // table 0 is for the initial line of quads
    static ui32 vlc_tbl0[2048];
    static ui32 vlc_tbl1[2048];

    //UVLC encoding
    static ui32 uvlc_tbl_pair1[33 * 33];
    static ui32 uvlc_tbl_pair2[33 * 33];
    static ui32 ulvc_cwd_pre[33];
    static int ulvc_cwd_pre_len[33];
    static ui32 ulvc_cwd_suf[33];
    static int ulvc_cwd_suf_len[33];

    /////////////////////////////////////////////////////////////////////////
    static bool vlc_init_tables()
    {
      struct vlc_src_table { int c_q, rho, u_off, e_k, e_1, cwd, cwd_len; };
      vlc_src_table tbl0[] = {
    #include "table0.h"
      };
      size_t tbl0_size = sizeof(tbl0) / sizeof(vlc_src_table);

      si32 pattern_popcnt[16];
      for (ui32 i = 0; i < 16; ++i)
        pattern_popcnt[i] = (si32)population_count(i);

      vlc_src_table* src_tbl = tbl0;
      ui32 *tgt_tbl = vlc_tbl0;
      size_t tbl_size = tbl0_size;
      for (int i = 0; i < 2048; ++i)
      {
        int c_q = i >> 8, rho = (i >> 4) & 0xF, emb = i & 0xF;
        if (((emb & rho) != emb) || (rho == 0 && c_q == 0))
          tgt_tbl[i] = 0;
        else
        {
          vlc_src_table *best_entry = NULL;
          if (emb) // u_off = 1
          {
            int best_e_k = -1;
            for (size_t j = 0; j < tbl_size; ++j)
            {
              if (src_tbl[j].c_q == c_q && src_tbl[j].rho == rho)
                if (src_tbl[j].u_off == 1)
                  if ((emb & src_tbl[j].e_k) == src_tbl[j].e_1)
                  {
                    //now we need to find the smallest cwd with the highest
                    // number of bits set in e_k
                    int ones_count = pattern_popcnt[src_tbl[j].e_k];
                    if (ones_count >= best_e_k)
                    {
                      best_entry = src_tbl + j;
                      best_e_k = ones_count;
                    }
                  }
            }
          }
          else // u_off = 0
          {
            for (size_t j = 0; j < tbl_size; ++j)
            {
              if (src_tbl[j].c_q == c_q && src_tbl[j].rho == rho)
                if (src_tbl[j].u_off == 0)
                {
                  best_entry = src_tbl + j;
                  break;
                }
            }
          }
          assert(best_entry);
          tgt_tbl[i] = (ui16)((best_entry->cwd<<8) + (best_entry->cwd_len<<4)
                             + best_entry->e_k);
        }
      }

      vlc_src_table tbl1[] = {
    #include "table1.h"
      };
      size_t tbl1_size = sizeof(tbl1) / sizeof(vlc_src_table);

      src_tbl = tbl1;
      tgt_tbl = vlc_tbl1;
      tbl_size = tbl1_size;
      for (int i = 0; i < 2048; ++i)
      {
        int c_q = i >> 8, rho = (i >> 4) & 0xF, emb = i & 0xF;
        if (((emb & rho) != emb) || (rho == 0 && c_q == 0))
          tgt_tbl[i] = 0;
        else
        {
          vlc_src_table *best_entry = NULL;
          if (emb) // u_off = 1
          {
            int best_e_k = -1;
            for (size_t j = 0; j < tbl_size; ++j)
            {
              if (src_tbl[j].c_q == c_q && src_tbl[j].rho == rho)
                if (src_tbl[j].u_off == 1)
                  if ((emb & src_tbl[j].e_k) == src_tbl[j].e_1)
                  {
                    //now we need to find the smallest cwd with the highest
                    // number of bits set in e_k
                    int ones_count = pattern_popcnt[src_tbl[j].e_k];
                    if (ones_count >= best_e_k)
                    {
                      best_entry = src_tbl + j;
                      best_e_k = ones_count;
                    }
                  }
            }
          }
          else // u_off = 0
          {
            for (size_t j = 0; j < tbl_size; ++j)
            {
              if (src_tbl[j].c_q == c_q && src_tbl[j].rho == rho)
                if (src_tbl[j].u_off == 0)
                {
                  best_entry = src_tbl + j;
                  break;
                }
            }
          }
          assert(best_entry);
          tgt_tbl[i] = (ui16)((best_entry->cwd<<8) + (best_entry->cwd_len<<4)
                             + best_entry->e_k);
        }
      }


      return true;
    }

    /////////////////////////////////////////////////////////////////////////
    static bool uvlc_init_tables()
    {
      //code goes from 0 to 31, extension and 32 are not supported here
      ulvc_cwd_pre[0] = 0; ulvc_cwd_pre[1] = 1; ulvc_cwd_pre[2] = 2;
      ulvc_cwd_pre[3] = 4; ulvc_cwd_pre[4] = 4;
      ulvc_cwd_pre_len[0] = 0; ulvc_cwd_pre_len[1] = 1;
      ulvc_cwd_pre_len[2] = 2;
      ulvc_cwd_pre_len[3] = 3; ulvc_cwd_pre_len[4] = 3;
      ulvc_cwd_suf[0] = 0; ulvc_cwd_suf[1] = 0; ulvc_cwd_suf[2] = 0;
      ulvc_cwd_suf[3] = 0; ulvc_cwd_suf[4] = 1;
      ulvc_cwd_suf_len[0] = 0; ulvc_cwd_suf_len[1] = 0;
      ulvc_cwd_suf_len[2] = 0;
      ulvc_cwd_suf_len[3] = 1; ulvc_cwd_suf_len[4] = 1;
      for (int i = 5; i < 33; ++i)
      {
        ulvc_cwd_pre[i] = 0;
        ulvc_cwd_pre_len[i] = 3;
        ulvc_cwd_suf[i] = (ui32)(i-5);
        ulvc_cwd_suf_len[i] = 5;
      }
      return true;
    }

    /////////////////////////////////////////////////////////////////////////
    static void uvlc_init_pair_tables()
    {
      for (int uq0 = 0; uq0 < 33; ++uq0) {
        for (int uq1 = 0; uq1 < 33; ++uq1) {
          ui32 cwd; int len;

          cwd = 0; len = 0;
          if (uq0 > 2 && uq1 > 2) {
            cwd |= ulvc_cwd_pre[uq0 - 2];
            len += ulvc_cwd_pre_len[uq0 - 2];
            cwd |= ulvc_cwd_pre[uq1 - 2] << len;
            len += ulvc_cwd_pre_len[uq1 - 2];
            cwd |= ulvc_cwd_suf[uq0 - 2] << len;
            len += ulvc_cwd_suf_len[uq0 - 2];
            cwd |= ulvc_cwd_suf[uq1 - 2] << len;
            len += ulvc_cwd_suf_len[uq1 - 2];
          } else if (uq0 > 2 && uq1 > 0) {
            cwd |= ulvc_cwd_pre[uq0];
            len += ulvc_cwd_pre_len[uq0];
            cwd |= (ui32)(uq1 - 1) << len;
            len += 1;
            cwd |= ulvc_cwd_suf[uq0] << len;
            len += ulvc_cwd_suf_len[uq0];
          } else {
            cwd |= ulvc_cwd_pre[uq0];
            len += ulvc_cwd_pre_len[uq0];
            cwd |= ulvc_cwd_pre[uq1] << len;
            len += ulvc_cwd_pre_len[uq1];
            cwd |= ulvc_cwd_suf[uq0] << len;
            len += ulvc_cwd_suf_len[uq0];
            cwd |= ulvc_cwd_suf[uq1] << len;
            len += ulvc_cwd_suf_len[uq1];
          }
          uvlc_tbl_pair1[uq0 * 33 + uq1] = (cwd << 5) | (ui32)len;

          cwd = 0; len = 0;
          cwd |= ulvc_cwd_pre[uq0];
          len += ulvc_cwd_pre_len[uq0];
          cwd |= ulvc_cwd_pre[uq1] << len;
          len += ulvc_cwd_pre_len[uq1];
          cwd |= ulvc_cwd_suf[uq0] << len;
          len += ulvc_cwd_suf_len[uq0];
          cwd |= ulvc_cwd_suf[uq1] << len;
          len += ulvc_cwd_suf_len[uq1];
          uvlc_tbl_pair2[uq0 * 33 + uq1] = (cwd << 5) | (ui32)len;
        }
      }
    }

    /////////////////////////////////////////////////////////////////////////
    bool initialize_block_encoder_tables_neon() {
      static bool tables_initialized = false;
      static std::once_flag tables_initialized_flag;
      std::call_once(tables_initialized_flag, []() {
        memset(vlc_tbl0, 0, 2048 * sizeof(ui32));
        memset(vlc_tbl1, 0, 2048 * sizeof(ui32));
        tables_initialized = vlc_init_tables();
        tables_initialized = tables_initialized && uvlc_init_tables();
        uvlc_init_pair_tables();
      });
      return tables_initialized;
    }

    /////////////////////////////////////////////////////////////////////////
    //
    /////////////////////////////////////////////////////////////////////////
    struct mel_struct {
      //storage
      ui8* buf;      //pointer to data buffer
      ui32 pos;      //position of next writing within buf
      ui32 buf_size; //size of buffer, which we must not exceed

      // all these can be replaced by bytes
      int remaining_bits; //number of empty bits in tmp
      int tmp;            //temporary storage of coded bits
      int run;            //number of 0 run
      int k;              //state
      int threshold;      //threshold where one bit must be coded
    };

    //////////////////////////////////////////////////////////////////////////
    static inline void
    mel_init(mel_struct* melp, ui32 buffer_size, ui8* data)
    {
      melp->buf = data;
      melp->pos = 0;
      melp->buf_size = buffer_size;
      melp->remaining_bits = 8;
      melp->tmp = 0;
      melp->run = 0;
      melp->k = 0;
      melp->threshold = 1; // this is 1 << mel_exp[melp->k];
    }

    static const int mel_exp[13] = {0,0,0,1,1,1,2,2,2,3,3,4,5};

    //////////////////////////////////////////////////////////////////////////
    static inline void
    mel_emit_bits(mel_struct* melp, ui32 bits, int num_bits)
    {
      melp->tmp = (melp->tmp << num_bits) | (int)bits;
      melp->remaining_bits -= num_bits;
      if (melp->remaining_bits <= 0) {
        int excess = -melp->remaining_bits;
        ui8 byte = (ui8)(melp->tmp >> excess);
        melp->buf[melp->pos++] = byte;
        melp->tmp &= (1 << excess) - 1;
        melp->remaining_bits += 8 - (byte == 0xFF);
      }
    }

    //////////////////////////////////////////////////////////////////////////
    static inline void
    mel_encode(mel_struct* melp, bool bit)
    {
      if (bit == false) {
        ++melp->run;
        if (melp->run >= melp->threshold) {
          mel_emit_bits(melp, 1, 1);
          melp->run = 0;
          melp->k = ojph_min(12, melp->k + 1);
          melp->threshold = 1 << mel_exp[melp->k];
        }
      } else {
        int t = mel_exp[melp->k];
        mel_emit_bits(melp, (ui32)melp->run & ((1u << t) - 1), t + 1);
        melp->run = 0;
        melp->k = ojph_max(0, melp->k - 1);
        melp->threshold = 1 << mel_exp[melp->k];
      }
    }

    //////////////////////////////////////////////////////////////////////////
    // static inline void
    // mel_advance_run(mel_struct* melp, ui32 n)
    // {
    //   ui32 remaining = n;
    //   while (remaining > 0) {
    //     ui32 space = (ui32)melp->threshold - (ui32)melp->run;
    //     if (remaining >= space) {
    //       remaining -= space;
    //       mel_emit_bits(melp, 1, 1);
    //       melp->run = 0;
    //       melp->k = ojph_min(12, melp->k + 1);
    //       melp->threshold = 1 << mel_exp[melp->k];
    //     } else {
    //       melp->run += (int)remaining;
    //       remaining = 0;
    //     }
    //   }
    // }

    //////////////////////////////////////////////////////////////////////////
    // static inline void
    // mel_encode_significance(mel_struct* melp)
    // {
    //   int t = mel_exp[melp->k];
    //   mel_emit_bits(melp, melp->run & ((1u << t) - 1), t + 1);
    //   melp->run = 0;
    //   melp->k = ojph_max(0, melp->k - 1);
    //   melp->threshold = 1 << mel_exp[melp->k];
    // }

    /////////////////////////////////////////////////////////////////////////
    //
    /////////////////////////////////////////////////////////////////////////

    struct vlc_struct {
      //storage
      ui8* buf;      //pointer to data buffer
      ui32 pos;      //position of next writing within buf
      ui32 buf_size; //size of buffer, which we must not exceed

      int used_bits; //number of occupied bits in tmp
      ui64 tmp;       //temporary storage of coded bits
      bool last_greater_than_8F; //true if last byte us greater than 0x8F
    };

    //////////////////////////////////////////////////////////////////////////
    static inline void
    vlc_init(vlc_struct* vlcp, ui32 buffer_size, ui8* data)
    {
      vlcp->buf = data + buffer_size - 1; //points to last byte
      vlcp->pos = 1;                      //locations will be all -pos
      vlcp->buf_size = buffer_size;

      vlcp->buf[0] = 0xFF;
      vlcp->used_bits = 4;
      vlcp->tmp = 0xF;
      vlcp->last_greater_than_8F = true;
    }

    //////////////////////////////////////////////////////////////////////////
    static inline void
    vlc_drain(vlc_struct* vlcp)
    {
      while (vlcp->used_bits >= 8) {
        int escape = (int)vlcp->last_greater_than_8F;
        int is_7f = (int)((vlcp->tmp & 0x7F) == 0x7F);
        int need_stuff = (escape & is_7f) != 0 ? 1 : 0;
        int bits = 8 - need_stuff;

        ui8 byte = (ui8)(vlcp->tmp & ((1u << bits) - 1));
        *(vlcp->buf - vlcp->pos) = byte;
        vlcp->pos++;
        vlcp->tmp >>= bits;
        vlcp->used_bits -= bits;
        vlcp->last_greater_than_8F = byte > 0x8F;
      }
    }

    //////////////////////////////////////////////////////////////////////////
    static inline void
    vlc_encode(vlc_struct* vlcp, ui64 cwd, int cwd_len)
    {
      while (true) {
        int avail = 64 - vlcp->used_bits;
        if (likely(avail > 0 && cwd_len <= avail)) {
          vlcp->tmp |= cwd << vlcp->used_bits;
          vlcp->used_bits += cwd_len;
          return;
        }
        if (likely(avail > 0)) // available space smaller than needed
          vlcp->tmp |= cwd << vlcp->used_bits;
        vlcp->used_bits = 64;
        vlc_drain(vlcp);
        cwd >>= avail;
        cwd_len -= avail;
      }
    }

    //////////////////////////////////////////////////////////////////////////
    //
    //////////////////////////////////////////////////////////////////////////
    static inline void
    terminate_mel_vlc(mel_struct* melp, vlc_struct* vlcp)
    {
      if (melp->run > 0)
        mel_emit_bits(melp, 1, 1);

      if (vlcp->last_greater_than_8F && (vlcp->tmp & 0x7f) == 0x7f) {
        *(vlcp->buf - vlcp->pos) = 0x7f;
        vlcp->pos++;
        vlcp->tmp >>= 7;
        vlcp->used_bits -= 7;
      }

      melp->tmp = melp->tmp << melp->remaining_bits;
      int mel_mask = (0xFF << melp->remaining_bits) & 0xFF;
      int vlc_mask = 0xFF >> (8 - vlcp->used_bits);
      if ((mel_mask | vlc_mask) == 0)
        return;  //last mel byte cannot be 0xFF, since then
                 //melp->remaining_bits would be < 8
      if (melp->pos >= melp->buf_size)
        OJPH_ERROR(0x00020003, "mel encoder's buffer is full");
      ui8 vlcp_tmp = (ui8)vlcp->tmp;
      int fuse = melp->tmp | vlcp_tmp;
      if ( ( ((fuse ^ melp->tmp) & mel_mask)
           | ((fuse ^ vlcp_tmp) & vlc_mask) ) == 0
          && (fuse != 0xFF) && vlcp->pos > 1)
      {
        melp->buf[melp->pos++] = (ui8)fuse;
      }
      else
      {
        if (vlcp->pos >= vlcp->buf_size)
          OJPH_ERROR(0x00020004, "vlc encoder's buffer is full");
        melp->buf[melp->pos++] = (ui8)melp->tmp; //melp->tmp cannot be 0xFF
        *(vlcp->buf - vlcp->pos) = (ui8)vlcp_tmp;
        vlcp->pos++;
      }
    }

/////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////

    struct ms_struct {
      //storage
      ui8* buf;        //pointer to data buffer
      ui32 pos;        //position of next writing within buf
      ui32 buf_size;   //size of buffer, which we must not exceed

      int used_bits;   //number of occupied bits in tmp
      ui64 tmp;        //temporary storage of coded bits (64-bit accumulator)
      bool last_was_ff;//true if the last written byte was 0xFF
    };

    //////////////////////////////////////////////////////////////////////////
    static inline void
    ms_init(ms_struct* msp, ui32 buffer_size, ui8* data)
    {
      msp->buf = data;
      msp->pos = 0;
      msp->buf_size = buffer_size;
      msp->used_bits = 0;
      msp->tmp = 0;
      msp->last_was_ff = false;
    }

    //////////////////////////////////////////////////////////////////////////
    static inline void
    ms_drain(ms_struct* msp)
    {
      if (msp->last_was_ff) {
        if (msp->used_bits < 7)
          return;
        msp->buf[msp->pos++] = (ui8)(msp->tmp & 0x7F);
        msp->tmp >>= 7;
        msp->used_bits -= 7;
        msp->last_was_ff = false;
      }

      while (msp->used_bits >= 8) {
        int n_bytes = msp->used_bits >> 3;
        if (n_bytes > 8) n_bytes = 8;

        ui64 word = msp->tmp;
        ui64 valid_mask = (n_bytes < 8)
                        ? (1ULL << (n_bytes * 8)) - 1 : ~(ui64)0;

        ui64 w = ~word;
        ui64 ff_detect = (w - 0x0101010101010101ULL) & ~w
                       & 0x8080808080808080ULL;
        ff_detect &= valid_mask;

        if (likely(ff_detect == 0)) {
          memcpy(msp->buf + msp->pos, &word, (size_t)n_bytes);
          msp->pos += (ui32)n_bytes;
          if (n_bytes < 8)
            msp->tmp >>= (n_bytes * 8);
          else
            msp->tmp = 0;
          msp->used_bits -= n_bytes * 8;
        } else {
          int ff_pos = (int)(count_trailing_zeros(ff_detect) >> 3);
          int safe = ff_pos + 1;
          memcpy(msp->buf + msp->pos, &word, (size_t)safe);
          msp->pos += (ui32)safe;
          int bits = safe * 8;
          if (bits < 64)
            msp->tmp >>= bits;
          else
            msp->tmp = 0;
          msp->used_bits -= bits;

          if (msp->used_bits >= 7) {
            msp->buf[msp->pos++] = (ui8)(msp->tmp & 0x7F);
            msp->tmp >>= 7;
            msp->used_bits -= 7;
            msp->last_was_ff = false;
          } else {
            msp->last_was_ff = true;
            return;
          }
        }
      }
    }

    //////////////////////////////////////////////////////////////////////////
    static inline void
    ms_encode_nodefer(ms_struct* msp, ui64 cwd, int cwd_len)
    {
      while (true) {
        int avail = 64 - msp->used_bits;
        if (likely(avail > 0 && cwd_len <= avail)) {
          msp->tmp |= cwd << msp->used_bits;
          msp->used_bits += cwd_len;
          return;
        }
        if (likely(avail > 0)) // available space smaller than needed
          msp->tmp |= cwd << msp->used_bits;
        msp->used_bits = 64;
        ms_drain(msp);
        cwd >>= avail;
        cwd_len -= avail;
      }
    }

    //////////////////////////////////////////////////////////////////////////
    // static inline void
    // ms_encode(ms_struct* msp, ui64 cwd, int cwd_len)
    // {
    //   int avail = 64 - msp->used_bits;
    //   if (likely(cwd_len <= avail)) {
    //     msp->tmp |= cwd << msp->used_bits;
    //     msp->used_bits += cwd_len;
    //   } else {
    //     msp->tmp |= (cwd & ((1ULL << avail) - 1)) << msp->used_bits;
    //     msp->used_bits = 64;
    //     ms_drain(msp);
    //     cwd >>= avail;
    //     cwd_len -= avail;
    //     msp->tmp |= cwd << msp->used_bits;
    //     msp->used_bits += cwd_len;
    //   }
    //   ms_drain(msp);
    // }

    //////////////////////////////////////////////////////////////////////////
    static inline void
    ms_terminate(ms_struct* msp)
    {
      ms_drain(msp);
      if (msp->used_bits)
      {
        int max_bits = msp->last_was_ff ? 7 : 8;
        int t = max_bits - msp->used_bits;
        ui32 byte = (ui32)(msp->tmp & ((1ULL << msp->used_bits) - 1));
        byte |= (0xFFu & ((1u << t) - 1)) << msp->used_bits;
        if (byte != 0xFF)
        {
          if (msp->pos >= msp->buf_size)
            OJPH_ERROR(0x00020006, "magnitude sign encoder's buffer is full");
          msp->buf[msp->pos++] = (ui8)byte;
        }
      }
      else if (msp->last_was_ff)
        msp->pos--;
    }


    //////////////////////////////////////////////////////////////////////////
    // 8 lanes of 32 bits as two NEON vectors: lanes 0..3 in lo, 4..7 in hi
    //////////////////////////////////////////////////////////////////////////
    struct v8u32 { uint32x4_t lo, hi; };

    static inline v8u32 v8_dup(ui32 v)
    { v8u32 r; r.lo = r.hi = vdupq_n_u32(v); return r; }
    static inline v8u32 v8_zero() { return v8_dup(0); }
    static inline v8u32 v8_load(const ui32 *p)
    { v8u32 r; r.lo = vld1q_u32(p); r.hi = vld1q_u32(p + 4); return r; }
    static inline void v8_store(ui32 *p, const v8u32 &a)
    { vst1q_u32(p, a.lo); vst1q_u32(p + 4, a.hi); }
    static inline v8u32 v8_add(const v8u32 &a, const v8u32 &b)
    { v8u32 r; r.lo = vaddq_u32(a.lo, b.lo); r.hi = vaddq_u32(a.hi, b.hi);
      return r; }
    static inline v8u32 v8_sub(const v8u32 &a, const v8u32 &b)
    { v8u32 r; r.lo = vsubq_u32(a.lo, b.lo); r.hi = vsubq_u32(a.hi, b.hi);
      return r; }
    static inline v8u32 v8_and(const v8u32 &a, const v8u32 &b)
    { v8u32 r; r.lo = vandq_u32(a.lo, b.lo); r.hi = vandq_u32(a.hi, b.hi);
      return r; }
    static inline v8u32 v8_or(const v8u32 &a, const v8u32 &b)
    { v8u32 r; r.lo = vorrq_u32(a.lo, b.lo); r.hi = vorrq_u32(a.hi, b.hi);
      return r; }
    // (~mask) & b
    static inline v8u32 v8_andnot(const v8u32 &mask, const v8u32 &b)
    { v8u32 r; r.lo = vbicq_u32(b.lo, mask.lo); r.hi = vbicq_u32(b.hi, mask.hi);
      return r; }
    static inline v8u32 v8_max_s(const v8u32 &a, const v8u32 &b)
    {
      v8u32 r;
      r.lo = vreinterpretq_u32_s32(vmaxq_s32(vreinterpretq_s32_u32(a.lo),
                                             vreinterpretq_s32_u32(b.lo)));
      r.hi = vreinterpretq_u32_s32(vmaxq_s32(vreinterpretq_s32_u32(a.hi),
                                             vreinterpretq_s32_u32(b.hi)));
      return r;
    }
    static inline v8u32 v8_min_s(const v8u32 &a, const v8u32 &b)
    {
      v8u32 r;
      r.lo = vreinterpretq_u32_s32(vminq_s32(vreinterpretq_s32_u32(a.lo),
                                             vreinterpretq_s32_u32(b.lo)));
      r.hi = vreinterpretq_u32_s32(vminq_s32(vreinterpretq_s32_u32(a.hi),
                                             vreinterpretq_s32_u32(b.hi)));
      return r;
    }
    // all ones where a > b (signed)
    static inline v8u32 v8_cmpgt_s(const v8u32 &a, const v8u32 &b)
    {
      v8u32 r;
      r.lo = vcgtq_s32(vreinterpretq_s32_u32(a.lo), vreinterpretq_s32_u32(b.lo));
      r.hi = vcgtq_s32(vreinterpretq_s32_u32(a.hi), vreinterpretq_s32_u32(b.hi));
      return r;
    }
    static inline v8u32 v8_cmpeq(const v8u32 &a, const v8u32 &b)
    { v8u32 r; r.lo = vceqq_u32(a.lo, b.lo); r.hi = vceqq_u32(a.hi, b.hi);
      return r; }
    // all ones where a != 0
    static inline v8u32 v8_nonzero(const v8u32 &a)
    { v8u32 r; r.lo = vtstq_u32(a.lo, a.lo); r.hi = vtstq_u32(a.hi, a.hi);
      return r; }
    // all ones where (a & b) != 0
    static inline v8u32 v8_test(const v8u32 &a, const v8u32 &b)
    { v8u32 r; r.lo = vtstq_u32(a.lo, b.lo); r.hi = vtstq_u32(a.hi, b.hi);
      return r; }
    template <int N> static inline v8u32 v8_srli(const v8u32 &a)
    { v8u32 r; r.lo = vshrq_n_u32(a.lo, N); r.hi = vshrq_n_u32(a.hi, N);
      return r; }
    template <int N> static inline v8u32 v8_slli(const v8u32 &a)
    { v8u32 r; r.lo = vshlq_n_u32(a.lo, N); r.hi = vshlq_n_u32(a.hi, N);
      return r; }
    // lane i <- lane i + 1 of the 16 lanes (a, b): a right shift by one lane
    // that takes b's lane 0 in at lane 7
    static inline v8u32 v8_next1(const v8u32 &a, const v8u32 &b)
    { v8u32 r; r.lo = vextq_u32(a.lo, a.hi, 1); r.hi = vextq_u32(a.hi, b.lo, 1);
      return r; }
    // lane i <- lane i + 2 of the 16 lanes (a, b)
    static inline v8u32 v8_next2(const v8u32 &a, const v8u32 &b)
    { v8u32 r; r.lo = vextq_u32(a.lo, a.hi, 2); r.hi = vextq_u32(a.hi, b.lo, 2);
      return r; }
    // lane i <- lane i - 1, lane 0 <- first: a left shift by one lane
    static inline v8u32 v8_prev1(ui32 first, const v8u32 &a)
    {
      v8u32 r;
      r.lo = vsetq_lane_u32(first, vextq_u32(a.hi, a.lo, 3), 0);
      r.hi = vextq_u32(a.lo, a.hi, 3);
      return r;
    }
    // the AVX2 encoder's rotation: lane i <- lane i + 1, lane 7 <- lane 0
    static inline v8u32 v8_rotr1(const v8u32 &a)
    { return v8_next1(a, a); }

    //////////////////////////////////////////////////////////////////////////
    static inline void
    proc_pixel(v8u32 *src_vec, ui32 p, v8u32 *eq_vec, v8u32 *s_vec,
               v8u32 &rho_vec, v8u32 &e_qmax_vec)
    {
      v8u32 val_vec[4];
      v8u32 _eq_vec[4];
      v8u32 _s_vec[4];
      v8u32 _rho_vec[4];
      const v8u32 one = v8_dup(1), thirty_two = v8_dup(32);
      const int32x4_t shift = vdupq_n_s32(-(int)p);

      for (ui32 i = 0; i < 4; ++i) {
        /* val = t + t; //multiply by 2 and get rid of sign */
        val_vec[i] = v8_add(src_vec[i], src_vec[i]);

        /* val >>= p;  // 2 \mu_p + x */
        val_vec[i].lo = vshlq_u32(val_vec[i].lo, shift);
        val_vec[i].hi = vshlq_u32(val_vec[i].hi, shift);

        /* val &= ~1u; // 2 \mu_p */
        val_vec[i] = v8_and(val_vec[i], v8_dup(~1u));

        /* if (val) { */
        const v8u32 val_notmask = v8_nonzero(val_vec[i]);

        /*   e_q[i] = 32 - (int)count_leading_zeros(--val); //2\mu_p - 1 */
        val_vec[i] = v8_sub(val_vec[i], one);
        v8u32 lz;
        lz.lo = vclzq_u32(val_vec[i].lo);
        lz.hi = vclzq_u32(val_vec[i].hi);
        _eq_vec[i] = v8_sub(thirty_two, lz);

        /*   s[0] = --val + (t >> 31); //v_n = 2(\mu_p-1) + s_n */
        val_vec[i] = v8_sub(val_vec[i], one);
        _s_vec[i] = v8_add(v8_srli<31>(src_vec[i]), val_vec[i]);

        _eq_vec[i] = v8_and(_eq_vec[i], val_notmask);
        _s_vec[i] = v8_and(_s_vec[i], val_notmask);
        val_vec[i] = v8_srli<31>(val_notmask);
        /* } */
      }

      // From  *_vec[0]: row 0, columns 0..7;  *_vec[1]: row 1, columns 0..7
      //       *_vec[2]: row 0, columns 8..15; *_vec[3]: row 1, columns 8..15
      // to    *_vec[0]: row 0, even columns;  *_vec[1]: row 1, even columns
      //       *_vec[2]: row 0, odd columns;   *_vec[3]: row 1, odd columns
      // (lane j of each is then quad j)
      for (ui32 i = 0; i < 2; ++i) {
        eq_vec[0 + i].lo = vuzp1q_u32(_eq_vec[0 + i].lo, _eq_vec[0 + i].hi);
        eq_vec[0 + i].hi = vuzp1q_u32(_eq_vec[2 + i].lo, _eq_vec[2 + i].hi);
        eq_vec[2 + i].lo = vuzp2q_u32(_eq_vec[0 + i].lo, _eq_vec[0 + i].hi);
        eq_vec[2 + i].hi = vuzp2q_u32(_eq_vec[2 + i].lo, _eq_vec[2 + i].hi);

        s_vec[0 + i].lo = vuzp1q_u32(_s_vec[0 + i].lo, _s_vec[0 + i].hi);
        s_vec[0 + i].hi = vuzp1q_u32(_s_vec[2 + i].lo, _s_vec[2 + i].hi);
        s_vec[2 + i].lo = vuzp2q_u32(_s_vec[0 + i].lo, _s_vec[0 + i].hi);
        s_vec[2 + i].hi = vuzp2q_u32(_s_vec[2 + i].lo, _s_vec[2 + i].hi);

        _rho_vec[0 + i].lo = vuzp1q_u32(val_vec[0 + i].lo, val_vec[0 + i].hi);
        _rho_vec[0 + i].hi = vuzp1q_u32(val_vec[2 + i].lo, val_vec[2 + i].hi);
        _rho_vec[2 + i].lo = vuzp2q_u32(val_vec[0 + i].lo, val_vec[0 + i].hi);
        _rho_vec[2 + i].hi = vuzp2q_u32(val_vec[2 + i].lo, val_vec[2 + i].hi);
      }

      e_qmax_vec = v8_max_s(eq_vec[0], eq_vec[1]);
      e_qmax_vec = v8_max_s(e_qmax_vec, eq_vec[2]);
      e_qmax_vec = v8_max_s(e_qmax_vec, eq_vec[3]);
      _rho_vec[1] = v8_slli<1>(_rho_vec[1]);
      _rho_vec[2] = v8_slli<2>(_rho_vec[2]);
      _rho_vec[3] = v8_slli<3>(_rho_vec[3]);
      rho_vec = v8_or(_rho_vec[0], _rho_vec[1]);
      rho_vec = v8_or(rho_vec, _rho_vec[2]);
      rho_vec = v8_or(rho_vec, _rho_vec[3]);
    }

    //////////////////////////////////////////////////////////////////////////
    // a 4 x 4 transpose of the rows r0..r3
    static inline void
    transpose4(uint32x4_t r0, uint32x4_t r1, uint32x4_t r2, uint32x4_t r3,
               uint32x4_t &c0, uint32x4_t &c1, uint32x4_t &c2, uint32x4_t &c3)
    {
      uint32x4_t t0 = vtrn1q_u32(r0, r1), t1 = vtrn2q_u32(r0, r1);
      uint32x4_t t2 = vtrn1q_u32(r2, r3), t3 = vtrn2q_u32(r2, r3);
      c0 = vreinterpretq_u32_u64(vtrn1q_u64(vreinterpretq_u64_u32(t0),
                                            vreinterpretq_u64_u32(t2)));
      c1 = vreinterpretq_u32_u64(vtrn1q_u64(vreinterpretq_u64_u32(t1),
                                            vreinterpretq_u64_u32(t3)));
      c2 = vreinterpretq_u32_u64(vtrn2q_u64(vreinterpretq_u64_u32(t0),
                                            vreinterpretq_u64_u32(t2)));
      c3 = vreinterpretq_u32_u64(vtrn2q_u64(vreinterpretq_u64_u32(t1),
                                            vreinterpretq_u64_u32(t3)));
    }

    // From matrix[k] lane j = sample k of quad j, to
    // matrix[k] = the 4 samples of quad 2k, then the 4 samples of quad 2k+1
    static inline void rotate_matrix(v8u32 *matrix)
    {
      uint32x4_t q0, q1, q2, q3, q4, q5, q6, q7;
      transpose4(matrix[0].lo, matrix[1].lo, matrix[2].lo, matrix[3].lo,
                 q0, q1, q2, q3);
      transpose4(matrix[0].hi, matrix[1].hi, matrix[2].hi, matrix[3].hi,
                 q4, q5, q6, q7);
      matrix[0].lo = q0; matrix[0].hi = q1;
      matrix[1].lo = q2; matrix[1].hi = q3;
      matrix[2].lo = q4; matrix[2].hi = q5;
      matrix[3].lo = q6; matrix[3].hi = q7;
    }

    //////////////////////////////////////////////////////////////////////////
    static inline void
    proc_ms_encode(ms_struct *msp, v8u32 &tuple_vec, v8u32 &uq_vec,
                   v8u32 &rho_vec, v8u32 *s_vec)
    {
      v8u32 m_vec[4];

      /* m = (rho[i] & (1 << k)) ? Uq[i] - ((tuple[i] >> k) & 1) : 0; */
      m_vec[0] = v8_and(v8_test(rho_vec, v8_dup(1)),
                        v8_sub(uq_vec, v8_and(tuple_vec, v8_dup(1))));
      m_vec[1] = v8_and(v8_test(rho_vec, v8_dup(2)),
        v8_sub(uq_vec, v8_srli<1>(v8_and(tuple_vec, v8_dup(2)))));
      m_vec[2] = v8_and(v8_test(rho_vec, v8_dup(4)),
        v8_sub(uq_vec, v8_srli<2>(v8_and(tuple_vec, v8_dup(4)))));
      m_vec[3] = v8_and(v8_test(rho_vec, v8_dup(8)),
        v8_sub(uq_vec, v8_srli<3>(v8_and(tuple_vec, v8_dup(8)))));

      rotate_matrix(m_vec);
      rotate_matrix(s_vec);

      ui32 cwd[8];
      int cwd_len[8];
      const uint32x4_t one = vdupq_n_u32(1);

      /* Each iteration process 8 bytes * 2 lines */
      for (ui32 i = 0; i < 4; ++i) {
        /* cwd = s[i * 4 + 0] & ((1U << m) - 1)
         * cwd_len = m
         */
        vst1q_s32(cwd_len, vreinterpretq_s32_u32(m_vec[i].lo));
        vst1q_s32(cwd_len + 4, vreinterpretq_s32_u32(m_vec[i].hi));
        // (a shift by 32 or more gives 0, as _mm256_sllv_epi32 does)
        uint32x4_t lo = vsubq_u32(
          vshlq_u32(one, vreinterpretq_s32_u32(m_vec[i].lo)), one);
        uint32x4_t hi = vsubq_u32(
          vshlq_u32(one, vreinterpretq_s32_u32(m_vec[i].hi)), one);
        vst1q_u32(cwd, vandq_u32(lo, s_vec[i].lo));
        vst1q_u32(cwd + 4, vandq_u32(hi, s_vec[i].hi));

        // A codeword of length 0 is 0, so where the bits before it fill
        // all 64 the shift is taken modulo 64 (a shift by 64 is undefined
        // in C++; the AVX2 encoder shifts by 64 there)
        for (ui32 j = 0; j < 4; j += 2) {
          ui32 idx0 = j * 2;
          ui64 _cwd     = cwd[idx0];
          int  _cwd_len = cwd_len[idx0];
          _cwd     |= ((ui64)cwd[idx0 + 1]) << _cwd_len;
          _cwd_len += cwd_len[idx0 + 1];

          ui32 idx1 = (j + 1) * 2;
          int len1 = cwd_len[idx1] + cwd_len[idx1 + 1];
          if (likely(_cwd_len + len1 <= 64)) {
            _cwd     |= ((ui64)cwd[idx1]) << (_cwd_len & 63);
            _cwd_len += cwd_len[idx1];
            _cwd     |= ((ui64)cwd[idx1 + 1]) << (_cwd_len & 63);
            _cwd_len += cwd_len[idx1 + 1];
            ms_encode_nodefer(msp, _cwd, _cwd_len);
          } else {
            ms_encode_nodefer(msp, _cwd, _cwd_len);
            _cwd     = cwd[idx1];
            _cwd_len = cwd_len[idx1];
            _cwd     |= ((ui64)cwd[idx1 + 1]) << _cwd_len;
            _cwd_len += cwd_len[idx1 + 1];
            ms_encode_nodefer(msp, _cwd, _cwd_len);
          }
        }
      }
      ms_drain(msp);
    }

    //////////////////////////////////////////////////////////////////////////
    static inline v8u32
    cal_eps_vec(v8u32 *eq_vec, v8u32 &u_q_vec, v8u32 &e_qmax_vec)
    {
      /* if (u_q[i] > 0) {
       *     eps[i] |= (e_q[i * 4 + k] == e_qmax[i]) << k;  k = 0..3
       * }
       */
      v8u32 u_q_mask = v8_cmpgt_s(u_q_vec, v8_zero());
      v8u32 eps_vec = v8_srli<31>(v8_cmpeq(eq_vec[0], e_qmax_vec));
      v8u32 t = v8_srli<31>(v8_cmpeq(eq_vec[1], e_qmax_vec));
      eps_vec = v8_or(eps_vec, v8_slli<1>(t));
      t = v8_srli<31>(v8_cmpeq(eq_vec[2], e_qmax_vec));
      eps_vec = v8_or(eps_vec, v8_slli<2>(t));
      t = v8_srli<31>(v8_cmpeq(eq_vec[3], e_qmax_vec));
      eps_vec = v8_or(eps_vec, v8_slli<3>(t));
      return v8_and(u_q_mask, eps_vec);
    }

    //////////////////////////////////////////////////////////////////////////
    static inline void
    update_lep(ui32 x, v8u32 &prev_e_val_vec, v8u32 *eq_vec,
               v8u32 *e_val_vec)
    {
      /* lep[0] = ojph_max(lep[0], (ui8)e_q[1]); lep++;
       * lep[0] = (ui8)e_q[3];
       * Compare e_q[1] with e_q[3] of the prevous round.
       */
      v8u32 t = v8_prev1(vgetq_lane_u32(prev_e_val_vec.lo, 0), eq_vec[3]);
      prev_e_val_vec = v8_zero();
      prev_e_val_vec.lo =
        vsetq_lane_u32(vgetq_lane_u32(eq_vec[3].hi, 3), prev_e_val_vec.lo, 0);
      e_val_vec[x] = v8_max_s(eq_vec[1], t);
    }

    //////////////////////////////////////////////////////////////////////////
    static inline void
    update_lcxp(ui32 x, v8u32 &prev_cx_val_vec, v8u32 &rho_vec,
                v8u32 *cx_val_vec)
    {
      /* lcxp[0] = (ui8)(lcxp[0] | (ui8)((rho[0] & 2) >> 1)); lcxp++;
       * lcxp[0] = (ui8)((rho[0] & 8) >> 3);
       * Or (rho[0] & 2) and (rho[0] of the previous round & 8).
       */
      v8u32 t = v8_prev1(vgetq_lane_u32(prev_cx_val_vec.lo, 0), rho_vec);
      prev_cx_val_vec = v8_zero();
      prev_cx_val_vec.lo =
        vsetq_lane_u32(vgetq_lane_u32(rho_vec.hi, 3), prev_cx_val_vec.lo, 0);

      t = v8_srli<3>(v8_and(t, v8_dup(8)));
      v8u32 t1 = v8_srli<1>(v8_and(rho_vec, v8_dup(2)));
      cx_val_vec[x] = v8_or(t, t1);
    }

    //////////////////////////////////////////////////////////////////////////
    static inline v8u32
    cal_tuple(v8u32 &cq_vec, v8u32 &rho_vec, v8u32 &eps_vec, ui32 *vlc_tbl)
    {
      /* tuple[i] = vlc_tbl1[(c_q[i] << 8) + (rho[i] << 4) + eps[i]]; */
      v8u32 t = v8_add(v8_add(v8_slli<8>(cq_vec), v8_slli<4>(rho_vec)),
                       eps_vec);
      ui32 idx[8], val[8];
      v8_store(idx, t);
      for (int i = 0; i < 8; ++i)
        val[i] = vlc_tbl[idx[i]];
      return v8_load(val);
    }

    //////////////////////////////////////////////////////////////////////////
    static inline v8u32
    proc_cq1(ui32 x, v8u32 *cx_val_vec, v8u32 &rho_vec)
    {
      ojph_unused(x);
      ojph_unused(cx_val_vec);
      /* c_q[i + 1] = (rho[i] >> 1) | (rho[i] & 1); */
      return v8_or(v8_srli<1>(rho_vec), v8_and(rho_vec, v8_dup(1)));
    }

    //////////////////////////////////////////////////////////////////////////
    static inline v8u32
    proc_cq2(ui32 x, v8u32 *cx_val_vec, v8u32 &rho_vec)
    {
      // c_q[i + 1] = (lcxp[i + 1] + (lcxp[i + 2] << 2))
      //            | (((rho[i] & 4) >> 1) | ((rho[i] & 8) >> 2));
      v8u32 t1 = v8_next1(cx_val_vec[x], cx_val_vec[x + 1]);
      v8u32 t = v8_slli<2>(v8_next2(cx_val_vec[x], cx_val_vec[x + 1]));
      t = v8_add(t1, t);
      t = v8_or(t, v8_srli<1>(v8_and(rho_vec, v8_dup(4))));
      return v8_or(t, v8_srli<2>(v8_and(rho_vec, v8_dup(8))));
    }

    //////////////////////////////////////////////////////////////////////////
    static inline void
    proc_mel_encode1(mel_struct *melp, v8u32 &cq_vec, v8u32 &rho_vec,
                     v8u32 u_q_vec, ui32 ignore)
    {
      ui32 mel_need_encode[8];
      ui32 mel_need_encode2[8];
      ui32 mel_bit[8];
      ui32 mel_bit2[8];
      /* Prepare mel_encode params */
      /* if (c_q[i] == 0) { */
      v8_store(mel_need_encode, v8_cmpeq(cq_vec, v8_zero()));
      /*   mel_encode(&mel, rho[i] != 0); */
      v8_store(mel_bit, v8_srli<31>(v8_nonzero(rho_vec)));
      /* } */

      /*   mel_encode(&mel, ojph_min(u_q[i], u_q[i + 1]) > 2); */
      v8u32 t = v8_rotr1(u_q_vec);
      v8u32 t1 = v8_min_s(u_q_vec, t);
      v8_store(mel_bit2, v8_srli<31>(v8_cmpgt_s(t1, v8_dup(2))));

      /* if (u_q[i] > 0 && u_q[i + 1] > 0) { } */
      v8u32 need_encode2 = v8_cmpgt_s(u_q_vec, v8_zero());
      v8_store(mel_need_encode2,
               v8_and(need_encode2, v8_cmpgt_s(t, v8_zero())));

      ui32 i_max = 8 - (ignore / 2);

      for (ui32 i = 0; i < i_max; i += 2) {
        if (mel_need_encode[i])
          mel_encode(melp, mel_bit[i] != 0);

        if (i + 1 < i_max) {
          if (mel_need_encode[i + 1])
            mel_encode(melp, mel_bit[i + 1] != 0);
        }

        if (mel_need_encode2[i])
          mel_encode(melp, mel_bit2[i] != 0);
      }
    }

    //////////////////////////////////////////////////////////////////////////
    static inline void
    proc_mel_encode2(mel_struct *melp, v8u32 &cq_vec, v8u32 &rho_vec,
                     v8u32 u_q_vec, ui32 ignore)
    {
      ojph_unused(u_q_vec);

      // one bit per quad whose c_q is 0, quad i at bit i
      static const ui32 bits_lo[4] = { 1, 2, 4, 8 };
      static const ui32 bits_hi[4] = { 16, 32, 64, 128 };
      v8u32 need = v8_cmpeq(cq_vec, v8_zero());
      ui32 mask = vaddvq_u32(vandq_u32(need.lo, vld1q_u32(bits_lo)))
                + vaddvq_u32(vandq_u32(need.hi, vld1q_u32(bits_hi)));

      ui32 i_max = 8 - (ignore / 2);
      if (i_max < 8)
        mask &= (1u << i_max) - 1;

      if (mask == 0)
        return;

      ui32 mel_bit[8];
      v8_store(mel_bit, v8_srli<31>(v8_nonzero(rho_vec)));

      while (mask) {
        ui32 i = (ui32)count_trailing_zeros(mask);
        mel_encode(melp, mel_bit[i] != 0);
        mask &= mask - 1;
      }
    }

    //////////////////////////////////////////////////////////////////////////
    static inline void
    build_vlc_uvlc_pair(ui32 *tuple, ui32 *u_q, ui32 i,
                        const ui32 *uvlc_tbl, ui64 &val, int &size)
    {
      val = tuple[i + 0] >> 4;
      size = tuple[i + 0] & 7;

      val |= (ui64)(tuple[i + 1] >> 4) << size;
      size += tuple[i + 1] & 7;

      ui32 entry = uvlc_tbl[u_q[i] * 33 + u_q[i + 1]];
      val |= (ui64)(entry >> 5) << size;
      size += entry & 0x1F;
    }

    //////////////////////////////////////////////////////////////////////////
    static inline void
    proc_vlc_encode(vlc_struct *vlcp, ui32 *tuple, ui32 *u_q, ui32 ignore,
                    const ui32 *uvlc_tbl)
    {
      ui32 i_max = 8 - (ignore / 2);

      ui32 i = 0;
      for (; i + 2 < i_max; i += 4) {
        ui64 val1; int size1;
        build_vlc_uvlc_pair(tuple, u_q, i, uvlc_tbl, val1, size1);
        ui64 val2; int size2;
        build_vlc_uvlc_pair(tuple, u_q, i + 2, uvlc_tbl, val2, size2);
        vlc_encode(vlcp, val1 | (val2 << size1), size1 + size2);
      }
      if (i < i_max) {
        ui64 val; int size;
        build_vlc_uvlc_pair(tuple, u_q, i, uvlc_tbl, val, size);
        vlc_encode(vlcp, val, size);
      }
    }

    //////////////////////////////////////////////////////////////////////////
    template<int PASS>
    OJPH_FORCE_INLINE void encode_x_loop(
      ui32 *sp, ui32 stride, ui32 height, ui32 y,
      ui32 n_loop, ui32 _width, ui32 ignore, ui32 p,
      mel_struct &mel, vlc_struct &vlc, ms_struct &ms,
      v8u32 *e_val_vec, v8u32 &prev_e_val_vec,
      v8u32 *cx_val_vec, v8u32 &prev_cx_val_vec,
      ui32 &prev_cq)
    {
      ui32 *vlc_tbl = (PASS == 1) ? vlc_tbl0 : vlc_tbl1;
      const v8u32 one = v8_dup(1);

      v8u32 eq_vec[4];
      v8u32 s_vec[4];
      v8u32 src_vec[4];

      /* 16 samples per iteration */
      for (ui32 x = 0; x < n_loop; ++x) {

        /* t = sp[i]; */
        if ((x == (n_loop - 1)) && (_width % 16)) {
          ui32 tmp_buf[16] = { 0 };
          memcpy(tmp_buf, sp, (_width % 16) * sizeof(ui32));
          src_vec[0] = v8_load(tmp_buf);
          src_vec[2] = v8_load(tmp_buf + 8);
          if (y + 1 < height) {
            memcpy(tmp_buf, sp + stride, (_width % 16) * sizeof(ui32));
            src_vec[1] = v8_load(tmp_buf);
            src_vec[3] = v8_load(tmp_buf + 8);
          }
          else {
            src_vec[1] = v8_zero();
            src_vec[3] = v8_zero();
          }
        }
        else {
          src_vec[0] = v8_load(sp);
          src_vec[2] = v8_load(sp + 8);

          if (y + 1 < height) {
            src_vec[1] = v8_load(sp + stride);
            src_vec[3] = v8_load(sp + 8 + stride);
          }
          else {
            src_vec[1] = v8_zero();
            src_vec[3] = v8_zero();
          }
          sp += 16;
        }

        v8u32 rho_vec, e_qmax_vec;
        proc_pixel(src_vec, p, eq_vec, s_vec, rho_vec, e_qmax_vec);

        // max_e[(i + 1) % num] = ojph_max(lep[i + 1], lep[i + 2]) - 1;
        v8u32 t = v8_next1(e_val_vec[x], e_val_vec[x + 1]);
        v8u32 max_e_vec = v8_sub(v8_max_s(t, e_val_vec[x]), one);

        // kappa[i] = (rho[i] & (rho[i] - 1)) ? ojph_max(1, max_e[i]) : 1;
        t = v8_max_s(max_e_vec, one);
        v8u32 t1 = v8_and(rho_vec, v8_sub(rho_vec, one));
        v8u32 cmp = v8_cmpeq(t1, v8_zero());
        v8u32 kappa_vec1_ = v8_and(cmp, one);
        v8u32 kappa_vec2_ = v8_andnot(cmp, t);
        const v8u32 kappa_vec = v8_max_s(kappa_vec1_, kappa_vec2_);

        if (PASS == 1)
          t = proc_cq1(x, cx_val_vec, rho_vec);
        else
          t = proc_cq2(x, cx_val_vec, rho_vec);

        v8u32 cq_vec = v8_prev1(prev_cq, t);
        prev_cq = vgetq_lane_u32(t.hi, 3);

        update_lep(x, prev_e_val_vec, eq_vec, e_val_vec);
        update_lcxp(x, prev_cx_val_vec, rho_vec, cx_val_vec);

        /* Uq[i] = ojph_max(e_qmax[i], kappa[i]); */
        /* u_q[i] = Uq[i] - kappa[i]; */
        v8u32 uq_vec = v8_max_s(kappa_vec, e_qmax_vec);
        v8u32 u_q_vec = v8_sub(uq_vec, kappa_vec);

        v8u32 eps_vec = cal_eps_vec(eq_vec, u_q_vec, e_qmax_vec);
        v8u32 tuple_vec = cal_tuple(cq_vec, rho_vec, eps_vec, vlc_tbl);
        ui32 _ignore = ((n_loop - 1) == x) ? ignore : 0;

        if (PASS == 1)
          proc_mel_encode1(&mel, cq_vec, rho_vec, u_q_vec, _ignore);
        else
          proc_mel_encode2(&mel, cq_vec, rho_vec, u_q_vec, _ignore);

        proc_ms_encode(&ms, tuple_vec, uq_vec, rho_vec, s_vec);

        ui32 u_q[10];
        ui32 tuple[10];
        tuple_vec = v8_srli<4>(tuple_vec);
        v8_store(tuple, tuple_vec);
        v8_store(u_q, u_q_vec);
        {
          ui32 i_max = 8 - (_ignore / 2);
          if (i_max & 1) { tuple[i_max] = 0; u_q[i_max] = 0; }
          tuple[8] = 0; u_q[8] = 0;
        }
        proc_vlc_encode(&vlc, tuple, u_q, _ignore,
            (PASS == 1) ? uvlc_tbl_pair1 : uvlc_tbl_pair2);
      }
    }

    //////////////////////////////////////////////////////////////////////////
    void ojph_encode_codeblock_neon(ui32* buf, ui32 missing_msbs,
                                    ui32 num_passes, ui32 _width,
                                    ui32 height, ui32 stride, ui32* lengths,
                                    ojph::mem_elastic_allocator *elastic,
                                    ojph::coded_lists *& coded)
    {
      ojph_unused(num_passes);                      //currently not used

      ui32 width = (_width + 15) & ~15u;
      ui32 ignore = width - _width;
      const int ms_size = (16384 * 16 + 14) / 15; //more than enough
      const int mel_vlc_size = 3072;              //more than enough
      const int mel_size = 192;
      const int vlc_size = mel_vlc_size - mel_size;

      ui8 ms_buf[ms_size];
      ui8 mel_vlc_buf[mel_vlc_size];
      ui8 *mel_buf = mel_vlc_buf;
      ui8 *vlc_buf = mel_vlc_buf + mel_size;

      mel_struct mel;
      mel_init(&mel, mel_size, mel_buf);
      vlc_struct vlc;
      vlc_init(&vlc, vlc_size, vlc_buf);
      ms_struct ms;
      ms_init(&ms, ms_size, ms_buf);

      const ui32 p = 30 - missing_msbs;

      //e_val: E values for a line (these are the highest set bit)
      //cx_val: is the context values
      //One lane per quad, as in the AVX2 encoder.
      ui32 n_loop = (width + 15) / 16;

      v8u32 e_val_vec[65];
      for (ui32 i = 0; i < ojph_min(64, n_loop); ++i)
        e_val_vec[i] = v8_zero();

      v8u32 prev_e_val_vec = v8_zero();

      v8u32 cx_val_vec[65];
      v8u32 prev_cx_val_vec = v8_zero();

      ui32 prev_cq = 0;

      /* 2 lines per iteration */
      for (ui32 y = 0; y < height; y += 2)
      {
        e_val_vec[n_loop] = prev_e_val_vec;
        /* lcxp[0] = (ui8)((rho[0] & 8) >> 3); */
        cx_val_vec[n_loop] = v8_srli<3>(v8_and(prev_cx_val_vec, v8_dup(8)));

        prev_e_val_vec = v8_zero();
        prev_cx_val_vec = v8_zero();

        ui32 *sp = buf + y * stride;

        if (y == 0)
          encode_x_loop<1>(sp, stride, height, y, n_loop, _width,
                           ignore, p, mel, vlc, ms,
                           e_val_vec, prev_e_val_vec,
                           cx_val_vec, prev_cx_val_vec, prev_cq);
        else
          encode_x_loop<2>(sp, stride, height, y, n_loop, _width,
                           ignore, p, mel, vlc, ms,
                           e_val_vec, prev_e_val_vec,
                           cx_val_vec, prev_cx_val_vec, prev_cq);

        // lane 0 of cx_val[0] + (lane 1 of cx_val[0] << 2)
        prev_cq = vgetq_lane_u32(cx_val_vec[0].lo, 0)
                + (vgetq_lane_u32(cx_val_vec[0].lo, 1) << 2);
      }

      ms_terminate(&ms);
      vlc_drain(&vlc);
      terminate_mel_vlc(&mel, &vlc);

      //copy to elastic
      lengths[0] = mel.pos + vlc.pos + ms.pos;
      elastic->get_buffer(mel.pos + vlc.pos + ms.pos, coded);
      memcpy(coded->buf, ms.buf, ms.pos);
      memcpy(coded->buf + ms.pos, mel.buf, mel.pos);
      memcpy(coded->buf + ms.pos + mel.pos, vlc.buf - vlc.pos + 1, vlc.pos);

      // put in the interface locator word
      ui32 num_bytes = mel.pos + vlc.pos;
      coded->buf[lengths[0]-1] = (ui8)(num_bytes >> 4);
      coded->buf[lengths[0]-2] = coded->buf[lengths[0]-2] & 0xF0;
      coded->buf[lengths[0]-2] =
          (ui8)(coded->buf[lengths[0]-2] | (num_bytes & 0xF));

      coded->avail_size -= lengths[0];
    }

  } /* namespace local */
} /* namespace ojph */

#endif // OJPH_ENABLE_NEON
