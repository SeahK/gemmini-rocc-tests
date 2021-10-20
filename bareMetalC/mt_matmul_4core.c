// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"
#include "util.h"

#define A_TRANSPOSE 0
#define B_TRANSPOSE 0
#define WARMUP 1
// in common
#define num_proc 4 // 2 for each
#define NO_BIAS true
#define REPEATING_BIAS 1
#define FULL_BIAS_WIDTH 1
#define dilation 1
#define OROW_DIVIDE ((int)(num_proc / 2))

#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#endif


// layer 1 config
#define MAT_DIM_I 128
#define MAT_DIM_J 320
#define MAT_DIM_K 320 // to avoid too significant bank conflict

//stride with software padding
//#define J_STRIDE (MAT_DIM_J % 128 == 0) ? (DIM_J + 64) : DIM_J

static elem_t inA1[MAT_DIM_I][MAT_DIM_K] row_align(MAX_BLOCK_LEN);
static elem_t inB1[MAT_DIM_K][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
static elem_t outC1[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
static ACC_T inD1[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN_ACC);

static elem_t inA2[MAT_DIM_I][MAT_DIM_K] row_align(MAX_BLOCK_LEN);
static elem_t inB2[MAT_DIM_K][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
static elem_t outC2[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
static ACC_T inD2[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN_ACC);

static elem_t inA3[MAT_DIM_I][MAT_DIM_K] row_align(MAX_BLOCK_LEN);
static elem_t inB3[MAT_DIM_K][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
static elem_t outC3[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
static ACC_T inD3[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN_ACC);

static elem_t inA4[MAT_DIM_I][MAT_DIM_K] row_align(MAX_BLOCK_LEN);
static elem_t inB4[MAT_DIM_K][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
static elem_t outC4[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
static ACC_T inD4[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN_ACC);


void thread_entry(int cid, int nc)
{
  gemmini_flush(0);
  for (int i = 0; i < nc; i++) {
    if (i == cid) printf("Thread %d/%d starting\n", cid, nc);
    barrier(nc);
  }
  
  //priority 3: high priority (for core 0 and 1) priority 2: low priority (for core 2 and 3)
  // warm up
  barrier(nc);
  uint64_t start = read_cycles();
  for(int j = 0; j < nc; j++){
     if(j==cid && j == 0) { 
			elem_t* A = (elem_t*) inA1;
			elem_t* B = (elem_t*) inB1;
			elem_t* C = (elem_t*) outC1;
			acc_t * D = (acc_t*) inD1;
			uint8_t priority =  3;

		 	tiled_matmul_auto_priority(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
					A, B, NO_BIAS ? NULL : D, C,
					MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
					MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
					NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
					false, false,
					false, !FULL_BIAS_WIDTH,
					3, priority,
					WS);

	  }
	  else if(j==cid && j == 1){
			elem_t* A = (elem_t*) inA2;
			elem_t* B = (elem_t*) inB2;
			elem_t* C = (elem_t*) outC2;
			acc_t * D = (acc_t*) inD2;
			uint8_t priority =  3;

		 	tiled_matmul_auto_priority(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
					A, B, NO_BIAS ? NULL : D, C,
					MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
					MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
					NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
					false, false,
					false, !FULL_BIAS_WIDTH,
					3, priority,
					WS);
	  }
	  else if(j==cid && j == 2){
			elem_t* A = (elem_t*) inA3;
			elem_t* B = (elem_t*) inB3;
			elem_t* C = (elem_t*) outC3;
			acc_t * D = (acc_t*) inD3;
			uint8_t priority = 2;// 2;

		 	tiled_matmul_auto_priority(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
					A, B, NO_BIAS ? NULL : D, C,
					MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
					MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
					NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
					false, false,
					false, !FULL_BIAS_WIDTH,
					3, priority,
					WS);
	  }
	  else if(j==cid && j == 3){
			elem_t* A = (elem_t*) inA4;
			elem_t* B = (elem_t*) inB4;
			elem_t* C = (elem_t*) outC4;
			acc_t * D = (acc_t*) inD4;
			uint8_t priority = 2;// 2;

		 	tiled_matmul_auto_priority(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
					A, B, NO_BIAS ? NULL : D, C,
					MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
					MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
					NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
					false, false,
					false, !FULL_BIAS_WIDTH,
					3, priority,
					WS);
	  }
  }


  uint64_t end = read_cycles();

  for(int i = 0; i < nc; i++){
     if (i == cid) {
		 printf("Thread %d warm-up Cycles taken: %llu\n", cid, end - start);
       const int total_macs = MAT_DIM_I * MAT_DIM_J * MAT_DIM_K;
       const int ideal_cycles = total_macs / (DIM * DIM);
       const int utilization = 100 * ideal_cycles / (end-start);
       printf("Utilization: %d%%\n", utilization);
	  }
  barrier(nc);
  }

  // after warm up
  barrier(nc);
  start = read_cycles();
  for(int j = 0; j < nc; j++){
     if(j==cid && j == 0) { 
			elem_t* A = (elem_t*) inA1;
			elem_t* B = (elem_t*) inB1;
			elem_t* C = (elem_t*) outC1;
			acc_t * D = (acc_t*) inD1;
			uint8_t priority = 3;
			tiled_matmul_auto_priority(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
					A, B, NO_BIAS ? NULL : D, C,
					MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
					MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
					NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
					false, false,
					false, !FULL_BIAS_WIDTH,
					3, priority,
					WS);

	  }
	  else if(j==cid && j == 1){
			elem_t* A = (elem_t*) inA2;
			elem_t* B = (elem_t*) inB2;
			elem_t* C = (elem_t*) outC2;
			acc_t * D = (acc_t*) inD2;
			uint8_t priority = 3;

		 	tiled_matmul_auto_priority(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
					A, B, NO_BIAS ? NULL : D, C,
					MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
					MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
					NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
					false, false,
					false, !FULL_BIAS_WIDTH,
					3, priority,
					WS);
	  }
	  else if(j==cid && j == 2){
			elem_t* A = (elem_t*) inA3;
			elem_t* B = (elem_t*) inB3;
			elem_t* C = (elem_t*) outC3;
			acc_t * D = (acc_t*) inD3;
			uint8_t priority = 2;//2;

		 	tiled_matmul_auto_priority(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
					A, B, NO_BIAS ? NULL : D, C,
					MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
					MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
					NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
					false, false,
					false, !FULL_BIAS_WIDTH,
					3, priority,
					WS);
	  }
	  else if(j==cid && j == 3){
			elem_t* A = (elem_t*) inA4;
			elem_t* B = (elem_t*) inB4;
			elem_t* C = (elem_t*) outC4;
			acc_t * D = (acc_t*) inD4;
			uint8_t priority = 2;//2;

		 	tiled_matmul_auto_priority(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
					A, B, NO_BIAS ? NULL : D, C,
					MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
					MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
					NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
					false, false,
					false, !FULL_BIAS_WIDTH,
					3, priority,
					WS);
	  }
  }


  end = read_cycles();

  for(int i = 0; i < nc; i++){
     if (i == cid) {
		 printf("Thread %d Cycles taken: %llu\n", cid, end - start);
       const int total_macs = MAT_DIM_I * MAT_DIM_J * MAT_DIM_K;
       const int ideal_cycles = total_macs / (DIM * DIM);
       const int utilization = 100 * ideal_cycles / (end-start);
       printf("Utilization: %d%%\n", utilization);
	  }
  barrier(nc);
  }
  exit(0);
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);
  exit(0);
}

