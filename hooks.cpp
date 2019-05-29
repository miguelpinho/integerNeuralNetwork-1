/* Copyright (c) 2006-2008 by Princeton University
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Princeton University nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY PRINCETON UNIVERSITY ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL PRINCETON UNIVERSITY BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** \file hooks.c
 * \brief An implementation of the PARSEC Hooks Instrumentation API.
 *
 * In this file the hooks library functions are implemented. The hooks API
 * is defined in header file hooks.h.
 *
 * The default functionality can be enabled and disabled by defining the
 * corresponding macros in file config.h. A detailed description of all
 * features is available there.
 */

/* NOTE: A detailed description of each hook function is available in file
 *       hooks.h.
 */

#include "hooks.h"
#include "config.h"

#include <stdint.h>
#include <stdio.h>
#include <assert.h>

#if ENABLE_TIMING
#include <sys/time.h>
/** \brief Time at beginning of execution of Region-of-Interest.
 *
 * This variable will store the time when the Region-of-Interest is entered.
 * time_begin and time_end allow an accurate measurement of the execution time
 * of the benchmark.
 *
 * Time measurement can be enabled in file config.h.
 */
static double time_begin;
/** \brief Time at end of execution of Region-of-Interest.
 *
 * This variable will store the time when the Region-of-Interest is left.
 * time_begin and time_end allow an accurate measurement of the execution time
 * of the benchmark.
 *
 * Time measurement can be enabled in file config.h.
 */
static double time_end;
#endif //ENABLE_TIMING

/** Enable debugging code */
#define DEBUG 0

#if DEBUG
/* Counters to keep track of number of invocations of hooks */
static int num_bench_begins = 0;
static int num_roi_begins = 0;
static int num_roi_ends = 0;
static int num_bench_ends = 0;
#endif

#if ENABLE_GEM5_HOOKS
//These functions allow to create checkpoints in the hooks during simulations with gem5
static void m5_checkpoint(uint64_t x, uint64_t y)
{
  register uint64_t x0 asm("x0") = x;
  register uint64_t x1 asm("x1") = y;
  asm volatile(".inst 0xff430110;" ::"r"(x0), "r"(x1));
};
static __attribute__((optimize("O0"))) void m5_exit(uint64_t x)
{
  register uint64_t x0 asm("x0") = x;
  asm volatile(".inst 0xff210110;" ::"r"(x0));
};
static __attribute__((optimize("O0"))) void m5_reset_stats(uint64_t x, uint64_t y)
{
  register uint64_t x0 asm("x0") = x;
  register uint64_t x1 asm("x1") = y;
  asm volatile(".inst 0xff400110;" ::"r"(x0), "r"(x1));
};
static __attribute__((optimize("O0"))) void m5_dump_stats(uint64_t x, uint64_t y)
{
  register uint64_t x0 asm("x0") = x;
  register uint64_t x1 asm("x1") = y;
  asm volatile(".inst 0xff410110;" ::"r"(x0), "r"(x1));
};
#endif

/** \brief Variable for unique identifier of workload.
 *
 * This variable stores the unique identifier of the current benchmark program.
 * It is set in function __parsec_bench_begin().
 */
static enum __parsec_benchmark bench;

/* NOTE: Please look at hooks.h to see how these functions are used */

void __parsec_bench_begin(enum __parsec_benchmark __bench)
{
#if DEBUG
  num_bench_begins++;
  assert(num_bench_begins == 1);
  assert(num_roi_begins == 0);
  assert(num_roi_ends == 0);
  assert(num_bench_ends == 0);
#endif //DEBUG

  printf(HOOKS_PREFIX " PARSEC Hooks Version " HOOKS_VERSION "\n");
  fflush(NULL);

  //Store global benchmark ID for other hook functions
  bench = __bench;
}

void __parsec_bench_end()
{
#if DEBUG
  num_bench_ends++;
  assert(num_bench_begins == 1);
  assert(num_roi_begins == 1);
  assert(num_roi_ends == 1);
  assert(num_bench_ends == 1);
#endif //DEBUG

  fflush(NULL);
#if ENABLE_TIMING
  printf(HOOKS_PREFIX " Total time spent in ROI: %.3fs\n", time_end - time_begin);
#endif //ENABLE_TIMING
  printf(HOOKS_PREFIX " Terminating\n");
}

void __parsec_roi_begin()
{
#if DEBUG
  num_roi_begins++;
  assert(num_bench_begins == 1);
  assert(num_roi_begins == 1);
  assert(num_roi_ends == 0);
  assert(num_bench_ends == 0);
#endif //DEBUG

  printf(HOOKS_PREFIX " Entering ROI\n");
  fflush(NULL);

#if ENABLE_TIMING
  struct timeval t;
  gettimeofday(&t, NULL);
  time_begin = (double)t.tv_sec + (double)t.tv_usec * 1e-6;
#endif //ENABLE_TIMING

#if ENABLE_GEM5_HOOKS
  printf(GEM5_HOOKS_PREFIX " Beginning of ROI\n");
  m5_checkpoint(0, 0);
  m5_reset_stats(0, 0);
#endif
}

void __parsec_roi_end()
{
#if DEBUG
  num_roi_ends++;
  assert(num_bench_begins == 1);
  assert(num_roi_begins == 1);
  assert(num_roi_ends == 1);
  assert(num_bench_ends == 0);
#endif //DEBUG

#if ENABLE_TIMING
  struct timeval t;
  gettimeofday(&t, NULL);
  time_end = (double)t.tv_sec + (double)t.tv_usec * 1e-6;
#endif //ENABLE_TIMING

#if ENABLE_GEM5_HOOKS
  m5_dump_stats(0, 0);
  printf(GEM5_HOOKS_PREFIX " Stats dumped.\n");
  printf(GEM5_HOOKS_PREFIX " Total time spent in ROI: %.3fs\n",
         time_end - time_begin);
#endif

  printf(HOOKS_PREFIX " Leaving ROI\n");
  fflush(NULL);
}
