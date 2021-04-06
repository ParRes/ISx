/*
Copyright (c) 2015, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products
      derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/
#include <shmem.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <string.h>
#include <unistd.h> // sleep()
#include <sys/stat.h>
#include <stdint.h>
#include "params.h"
#include "isx.h"
#include "timer.h"
#include "pcg_basic.h"

#define ROOT_PE 0

// Needed for shmem collective operations
int pWrk[_SHMEM_REDUCE_MIN_WRKDATA_SIZE];
double dWrk[_SHMEM_REDUCE_SYNC_SIZE];
long long int llWrk[_SHMEM_REDUCE_MIN_WRKDATA_SIZE];
long pSync[_SHMEM_REDUCE_SYNC_SIZE];

uint64_t NUM_PES; // Number of parallel workers
uint64_t TOTAL_KEYS; // Total number of keys across all PEs
uint64_t NUM_KEYS_PER_PE; // Number of keys generated on each PE
uint64_t NUM_BUCKETS; // The number of buckets in the bucket sort
uint64_t BUCKET_WIDTH; // The size of each bucket
uint64_t MAX_KEY_VAL; // The maximum possible generated key value
uint64_t NUM_ITERATIONS; // Number of iterations that the sort is performed

volatile int whose_turn;

long long int receive_offset = 0;
long long int my_bucket_size = 0;


#define KEY_BUFFER_SIZE (1uLL<<28uLL)

// The receive array for the All2All exchange
KEY_TYPE *my_bucket_keys;

#ifdef PERMUTE
int * permute_array;
#endif

int main(const int argc,  char ** argv)
{
  shmem_init();
#ifdef OPENSHMEM_COMPLIANT
  my_bucket_keys = (KEY_TYPE*) shmem_malloc(KEY_BUFFER_SIZE * sizeof(KEY_TYPE));
#else
  my_bucket_keys = (KEY_TYPE*) shmalloc(KEY_BUFFER_SIZE * sizeof(KEY_TYPE));
#endif


  init_shmem_sync_array(pSync);

  char * log_file = parse_params(argc, argv);

  int err = bucket_sort();

  log_times(log_file);

  shmem_finalize();
  return err;
}


// Parses all of the command line input and definitions in params.h
// to set all necessary runtime values and options
static char * parse_params(const int argc, char ** argv)
{

  NUM_PES = (uint64_t) shmem_n_pes();
  MAX_KEY_VAL = DEFAULT_MAX_KEY;
  NUM_BUCKETS = NUM_PES;
  BUCKET_WIDTH = (uint64_t) ceil((double)MAX_KEY_VAL/NUM_BUCKETS);
  char scaling_msg[64];
  char * log_file;

  if(argc == 3) {
    NUM_ITERATIONS = 1u;
    log_file = argv[2];
  }
  else if(argc == 4) {
    NUM_ITERATIONS = (uint64_t) strtoull(argv[2], NULL, 10);
    log_file = argv[3];
  }
  else {
    if( shmem_my_pe() == 0){
      printf("Usage:  \n");
      printf("  ./%s <total num keys(strong) | keys per pe(weak)> [iterations] "
             "<log_file>\n",argv[0]);
    }

    shmem_finalize();
    exit(1);
  }

  switch(SCALING_OPTION){
    case STRONG:
      {
        TOTAL_KEYS = (uint64_t) strtoull(argv[1], NULL, 10);
        NUM_KEYS_PER_PE = (uint64_t) ceil((double)TOTAL_KEYS/NUM_PES);
        sprintf(scaling_msg,"STRONG");
        break;
      }

    case WEAK:
      {
        NUM_KEYS_PER_PE = (uint64_t) (strtoull(argv[1], NULL, 10));
        sprintf(scaling_msg,"WEAK");
        break;
      }

    case WEAK_ISOBUCKET:
      {
        NUM_KEYS_PER_PE = (uint64_t) (strtoull(argv[1], NULL, 10));
        BUCKET_WIDTH = ISO_BUCKET_WIDTH;
        MAX_KEY_VAL = (uint64_t) (NUM_PES * BUCKET_WIDTH);
        sprintf(scaling_msg,"WEAK_ISOBUCKET");
        break;
      }

    default:
      {
        if(shmem_my_pe() == 0){
          printf("Invalid scaling option! See params.h to define the scaling option.\n");
        }

        shmem_finalize();
        exit(1);
        break;
      }
  }

  assert(MAX_KEY_VAL > 0);
  assert(NUM_KEYS_PER_PE > 0);
  assert(NUM_PES > 0);
  assert(MAX_KEY_VAL > NUM_PES);
  assert(NUM_BUCKETS > 0);
  assert(BUCKET_WIDTH > 0);

  if(shmem_my_pe() == 0){
    printf("ISx v%1d.%1d\n",MAJOR_VERSION_NUMBER,MINOR_VERSION_NUMBER);
#ifdef PERMUTE
    printf("Random Permute Used in ATA.\n");
#endif
    printf("  Number of Keys per PE: %" PRIu64 "\n", NUM_KEYS_PER_PE);
    printf("  Max Key Value: %" PRIu64 "\n", MAX_KEY_VAL);
    printf("  Bucket Width: %" PRIu64 "\n", BUCKET_WIDTH);
    printf("  Number of Iterations: %" PRIu64 "\n", NUM_ITERATIONS);
    printf("  Number of PEs: %" PRIu64 "\n", NUM_PES);
    printf("  %s Scaling!\n",scaling_msg);
    }

  return log_file;
}


/*
 * The primary compute function for the bucket sort
 * Executes the sum of NUM_ITERATIONS + BURN_IN iterations, as defined in params.h
 * Only iterations after the BURN_IN iterations are timed
 * Only the final iteration calls the verification function
 */
static int bucket_sort(void)
{
  int err = 0;

  init_timers(NUM_ITERATIONS);

#ifdef PERMUTE
  create_permutation_array();
#endif

  for(uint64_t i = 0; i < (NUM_ITERATIONS + BURN_IN); ++i)
  {

    // Reset timers after burn in
    if(i == BURN_IN){ init_timers(NUM_ITERATIONS); }

    shmem_barrier_all();

    timer_start(&timers[TIMER_TOTAL]);

    KEY_TYPE * my_keys = make_input();

    int * local_bucket_sizes = count_local_bucket_sizes(my_keys);

    int * send_offsets;
    int * local_bucket_offsets = compute_local_bucket_offsets(local_bucket_sizes,
                                                                   &send_offsets);

    KEY_TYPE * my_local_bucketed_keys =  bucketize_local_keys(my_keys, local_bucket_offsets);

    KEY_TYPE * my_bucket_keys = exchange_keys(send_offsets,
                                              local_bucket_sizes,
                                              my_local_bucketed_keys);

    my_bucket_size = receive_offset;

    int * my_local_key_counts = count_local_keys(my_bucket_keys);

    shmem_barrier_all();

    timer_stop(&timers[TIMER_TOTAL]);

    // Only the last iteration is verified
    if(i == NUM_ITERATIONS) {
      err = verify_results(my_local_key_counts, my_bucket_keys);
    }

    // Reset receive_offset used in exchange_keys
    receive_offset = 0;

    free(my_local_bucketed_keys);
    free(my_keys);
    free(local_bucket_sizes);
    free(local_bucket_offsets);
    free(send_offsets);
    free(my_local_key_counts);

    shmem_barrier_all();
  }

#ifdef OPENSHMEM_COMPLIANT
  shmem_free(my_bucket_keys);
#else
  shfree(my_bucket_keys);
#endif

  return err;
}


/*
 * Generates uniformly random keys [0, MAX_KEY_VAL] on each rank using the time and rank
 * number as a seed
 */
static KEY_TYPE * make_input(void)
{
  timer_start(&timers[TIMER_INPUT]);

  KEY_TYPE * restrict const my_keys = malloc(NUM_KEYS_PER_PE * sizeof(KEY_TYPE));

  pcg32_random_t rng = seed_my_rank();

  for(uint64_t i = 0; i < NUM_KEYS_PER_PE; ++i) {
    my_keys[i] = pcg32_boundedrand_r(&rng, MAX_KEY_VAL);
  }

  timer_stop(&timers[TIMER_INPUT]);

#ifdef DEBUG
  wait_my_turn();
  char msg[1024];
  const int my_rank = shmem_my_pe();
  sprintf(msg,"Rank %d: Initial Keys: ", my_rank);
  for(uint64_t i = 0; i < NUM_KEYS_PER_PE; ++i){
    if(i < PRINT_MAX)
    sprintf(msg + strlen(msg),"%d ", my_keys[i]);
  }
  sprintf(msg + strlen(msg),"\n");
  printf("%s",msg);
  fflush(stdout);
  my_turn_complete();
#endif
  return my_keys;
}


/*
 * Computes the size of each bucket by iterating all keys and incrementing
 * their corresponding bucket's size
 */
static inline int * count_local_bucket_sizes(KEY_TYPE const * restrict const my_keys)
{
  int * restrict const local_bucket_sizes = malloc(NUM_BUCKETS * sizeof(int));

  timer_start(&timers[TIMER_BCOUNT]);

  init_array(local_bucket_sizes, NUM_BUCKETS);

  for(uint64_t i = 0; i < NUM_KEYS_PER_PE; ++i){
    const uint32_t bucket_index = my_keys[i]/BUCKET_WIDTH;
    local_bucket_sizes[bucket_index]++;
  }

  timer_stop(&timers[TIMER_BCOUNT]);

#ifdef DEBUG
  wait_my_turn();
  char msg[1024];
  const int my_rank = shmem_my_pe();
  sprintf(msg,"Rank %d: local bucket sizes: ", my_rank);
  for(uint64_t i = 0; i < NUM_BUCKETS; ++i){
    if(i < PRINT_MAX)
    sprintf(msg + strlen(msg),"%d ", local_bucket_sizes[i]);
  }
  sprintf(msg + strlen(msg),"\n");
  printf("%s",msg);
  fflush(stdout);
  my_turn_complete();
#endif

  return local_bucket_sizes;
}


/*
 * Computes the prefix scan of the bucket sizes to determine the starting locations
 * of each bucket in the local bucketed array
 * Stores a copy of the bucket offsets for use in exchanging keys because the
 * original bucket_offsets array is modified in the bucketize function
 */
static inline int * compute_local_bucket_offsets(int const * restrict const local_bucket_sizes,
                                                 int ** restrict send_offsets)
{
  int * restrict const local_bucket_offsets = malloc(NUM_BUCKETS * sizeof(int));

  timer_start(&timers[TIMER_BOFFSET]);

  (*send_offsets) = malloc(NUM_BUCKETS * sizeof(int));

  local_bucket_offsets[0] = 0;
  (*send_offsets)[0] = 0;
  int temp = 0;
  for(uint64_t i = 1; i < NUM_BUCKETS; i++){
    temp = local_bucket_offsets[i-1] + local_bucket_sizes[i-1];
    local_bucket_offsets[i] = temp;
    (*send_offsets)[i] = temp;
  }
  timer_stop(&timers[TIMER_BOFFSET]);

#ifdef DEBUG
  wait_my_turn();
  char msg[1024];
  const int my_rank = shmem_my_pe();
  sprintf(msg,"Rank %d: local bucket offsets: ", my_rank);
  for(uint64_t i = 0; i < NUM_BUCKETS; ++i){
    if(i < PRINT_MAX)
    sprintf(msg + strlen(msg),"%d ", local_bucket_offsets[i]);
  }
  sprintf(msg + strlen(msg),"\n");
  printf("%s",msg);
  fflush(stdout);
  my_turn_complete();
#endif
  return local_bucket_offsets;
}

/*
 * Places local keys into their corresponding local bucket.
 * The contents of each bucket are not sorted.
 */
static inline KEY_TYPE * bucketize_local_keys(KEY_TYPE const * restrict const my_keys,
                                              int * restrict const local_bucket_offsets)
{
  KEY_TYPE * restrict const my_local_bucketed_keys = malloc(NUM_KEYS_PER_PE * sizeof(KEY_TYPE));

  timer_start(&timers[TIMER_BUCKETIZE]);

  for(uint64_t i = 0; i < NUM_KEYS_PER_PE; ++i){
    const KEY_TYPE key = my_keys[i];
    const uint32_t bucket_index = key / BUCKET_WIDTH;
    uint32_t index;
    assert(local_bucket_offsets[bucket_index] >= 0);
    index = local_bucket_offsets[bucket_index]++;
    assert(index < NUM_KEYS_PER_PE);
    my_local_bucketed_keys[index] = key;
  }

  timer_stop(&timers[TIMER_BUCKETIZE]);

#ifdef DEBUG
  wait_my_turn();
  char msg[1024];
  const int my_rank = shmem_my_pe();
  sprintf(msg,"Rank %d: local bucketed keys: ", my_rank);
  for(uint64_t i = 0; i < NUM_KEYS_PER_PE; ++i){
    if(i < PRINT_MAX)
    sprintf(msg + strlen(msg),"%d ", my_local_bucketed_keys[i]);
  }
  sprintf(msg + strlen(msg),"\n");
  printf("%s",msg);
  fflush(stdout);
  my_turn_complete();
#endif
  return my_local_bucketed_keys;
}


/*
 * Each PE sends the contents of its local buckets to the PE that owns that bucket.
 */
static inline KEY_TYPE * exchange_keys(int const * restrict const send_offsets,
                                       int const * restrict const local_bucket_sizes,
                                       KEY_TYPE const * restrict const my_local_bucketed_keys)
{
  timer_start(&timers[TIMER_ATA_KEYS]);

  const int my_rank = shmem_my_pe();
  unsigned int total_keys_sent = 0;

  // Keys destined for local key buffer can be written with memcpy
  const long long int write_offset_into_self = shmem_longlong_fadd(&receive_offset, (long long int)local_bucket_sizes[my_rank], my_rank);
  memcpy(&my_bucket_keys[write_offset_into_self],
         &my_local_bucketed_keys[send_offsets[my_rank]],
         local_bucket_sizes[my_rank]*sizeof(KEY_TYPE));


  for(uint64_t i = 0; i < NUM_PES; ++i){

#ifdef PERMUTE
    const int target_pe = permute_array[i];
#elif INCAST
    const int target_pe = i;
#else
    const int target_pe = (my_rank + i) % NUM_PES;
#endif

    // Local keys already written with memcpy
    if(target_pe == my_rank){ continue; }

    const int read_offset_from_self = send_offsets[target_pe];
    const int my_send_size = local_bucket_sizes[target_pe];

    const long long int write_offset_into_target = shmem_longlong_fadd(&receive_offset, (long long int)my_send_size, target_pe);

#ifdef DEBUG
    printf("Rank: %d Target: %d Offset into target: %lld Offset into myself: %d Send Size: %d\n",
        my_rank, target_pe, write_offset_into_target, read_offset_from_self, my_send_size);
#endif

    shmem_int_put(&(my_bucket_keys[write_offset_into_target]),
                  &(my_local_bucketed_keys[read_offset_from_self]),
                  my_send_size,
                  target_pe);

    total_keys_sent += my_send_size;
  }

#ifdef BARRIER_ATA
  shmem_barrier_all();
#endif

  timer_stop(&timers[TIMER_ATA_KEYS]);
  timer_count(&timers[TIMER_ATA_KEYS], total_keys_sent);

#ifdef DEBUG
  wait_my_turn();
  char msg[1024];
  sprintf(msg,"Rank %d: Bucket Size %lld | Total Keys Sent: %u | Keys after exchange:",
                        my_rank, receive_offset, total_keys_sent);
  for(long long int i = 0; i < receive_offset; ++i){
    if(i < PRINT_MAX)
    sprintf(msg + strlen(msg),"%d ", my_bucket_keys[i]);
  }
  sprintf(msg + strlen(msg),"\n");
  printf("%s",msg);
  fflush(stdout);
  my_turn_complete();
#endif

  return my_bucket_keys;
}


/*
 * Counts the occurence of each key in my bucket.
 * Key indices into the count array are the key's value minus my bucket's
 * minimum key value to allow indexing from 0.
 * my_bucket_keys: All keys in my bucket unsorted [my_rank * BUCKET_WIDTH, (my_rank+1)*BUCKET_WIDTH)
 */
static inline int * count_local_keys(KEY_TYPE const * restrict const my_bucket_keys)
{
  int * restrict const my_local_key_counts = malloc(BUCKET_WIDTH * sizeof(int));
  memset(my_local_key_counts, 0, BUCKET_WIDTH * sizeof(int));

  timer_start(&timers[TIMER_SORT]);

  const int my_rank = shmem_my_pe();
  const int my_min_key = my_rank * BUCKET_WIDTH;

  // Count the occurences of each key in my bucket
  for(long long int i = 0; i < my_bucket_size; ++i){
    const unsigned int key_index = my_bucket_keys[i] - my_min_key;

    assert(my_bucket_keys[i] >= my_min_key);
    assert(key_index < BUCKET_WIDTH);

    my_local_key_counts[key_index]++;
  }
  timer_stop(&timers[TIMER_SORT]);

#ifdef DEBUG
  wait_my_turn();
  char msg[4096];
  sprintf(msg,"Rank %d: Bucket Size %lld | Local Key Counts:", my_rank, my_bucket_size);
  for(uint64_t i = 0; i < BUCKET_WIDTH; ++i){
    if(i < PRINT_MAX)
    sprintf(msg + strlen(msg),"%d ", my_local_key_counts[i]);
  }
  sprintf(msg + strlen(msg),"\n");
  printf("%s",msg);
  fflush(stdout);
  my_turn_complete();
#endif

  return my_local_key_counts;
}

/*
 * Verifies the correctness of the sort.
 * Ensures all keys are within a PE's bucket boundaries.
 * Ensures the final number of keys is equal to the initial.
 */
static int verify_results(int const * restrict const my_local_key_counts,
                           KEY_TYPE const * restrict const my_local_keys)
{

  shmem_barrier_all();

  int error = 0;

  const int my_rank = shmem_my_pe();

  const int my_min_key = my_rank * BUCKET_WIDTH;
  const int my_max_key = (my_rank+1) * BUCKET_WIDTH - 1;

  // Verify all keys are within bucket boundaries
  for(long long int i = 0; i < my_bucket_size; ++i){
    const int key = my_local_keys[i];
    if((key < my_min_key) || (key > my_max_key)){
      printf("Rank %d Failed Verification!\n",my_rank);
      printf("Key: %d is outside of bounds [%d, %d]\n", key, my_min_key, my_max_key);
      error = 1;
    }
  }

  // Verify the sum of the key population equals the expected bucket size
  long long int bucket_size_test = 0;
  for(uint64_t i = 0; i < BUCKET_WIDTH; ++i){
    bucket_size_test +=  my_local_key_counts[i];
  }
  if(bucket_size_test != my_bucket_size){
      printf("Rank %d Failed Verification!\n",my_rank);
      printf("Actual Bucket Size: %lld Should be %lld\n", bucket_size_test, my_bucket_size);
      error = 1;
  }

  // Verify the final number of keys equals the initial number of keys
  static long long int total_num_keys = 0;
  shmem_longlong_sum_to_all(&total_num_keys, &my_bucket_size, 1, 0, 0, NUM_PES, llWrk, pSync);
  shmem_barrier_all();

  if(total_num_keys != (long long int)(NUM_KEYS_PER_PE * NUM_PES)){
    if(my_rank == ROOT_PE){
      printf("Verification Failed!\n");
      printf("Actual total number of keys: %lld Expected %" PRIu64 "\n", total_num_keys, NUM_KEYS_PER_PE * NUM_PES );
      error = 1;
    }
  }

  return error;
}

/*
 * Gathers all the timing information from each PE and prints
 * it to a file. All information from a PE is printed as a row in a tab seperated file
 */
static void log_times(char * log_file)
{
  FILE * fp = NULL;

  if(shmem_my_pe() == ROOT_PE)
  {
    int print_names = 0;
    if(file_exists(log_file) != 1){
      print_names = 1;
    }

    if((fp = fopen(log_file, "a+b"))==NULL){
      perror("Error opening log file:");
      exit(1);
    }

    if(print_names == 1){
      print_run_info(fp);
      print_timer_names(fp);
    }
    fclose(fp);
  }

  for(int i = 0; i < shmem_n_pes(); ++i) {
    if (i == shmem_my_pe()) {
      if((fp = fopen(log_file, "a+b")) == NULL) {
        perror("Error opening log file:");
        exit(1);
      }

      print_timer_values(fp);
      fclose(fp);
    }
    shmem_barrier_all();
  }

  for(uint64_t t = 0; t < TIMER_NTIMERS; ++t){
    timers[t].pe_average_times = gather_rank_times(&timers[t]);
    // No need gather the average counts since we are not currently reporting them
    //timers[t].pe_average_counts = gather_rank_counts(&timers[t]);
  }

  if(shmem_my_pe() == ROOT_PE)
  {
    report_summary_stats();
  }
}

/*
 * Computes the average total time and average all2all time and prints it to the command line
 */
static void report_summary_stats(void)
{
  // We're exploiting the fact that each PE has the same number of iterations,
  // so the average of the averages across all PEs is equal to the average of
  // the entire collection. However, this would no longer be true if the PEs had
  // a different number of iterations.

  if(timers[TIMER_TOTAL].seconds_iter > 0) {
    double temp = 0.0;
    for(uint64_t i = 0; i < NUM_PES; ++i){
      temp += timers[TIMER_TOTAL].pe_average_times[i];
    }
    printf("Average total time (per PE): %f seconds\n", temp/NUM_PES);
  }

  if(timers[TIMER_ATA_KEYS].seconds_iter >0) {
    double temp = 0.0;
    for(uint64_t i = 0; i < NUM_PES; ++i){
      temp += timers[TIMER_ATA_KEYS].pe_average_times[i];
    }
    printf("Average all2all time (per PE): %f seconds\n", temp/NUM_PES);
  }
}

/*
 * Prints all the labels for each timer as a row to the file specified by 'fp'
 */
static void print_timer_names(FILE * fp)
{
  for(uint64_t i = 0; i < TIMER_NTIMERS; ++i){
    if(timers[i].seconds_iter > 0){
      fprintf(fp, "%s (sec)\t", timer_names[i]);
    }
    if(timers[i].count_iter > 0){
      fprintf(fp, "%s_COUNTS\t", timer_names[i]);
    }
  }
  fprintf(fp,"\n");
}

/*
 * Prints all the relevant runtime parameters as a row to the file specified by 'fp'
 */
static void print_run_info(FILE * fp)
{
  fprintf(fp,"SHMEM\t");
  fprintf(fp,"NUM_PES %" PRIu64 "\t", NUM_PES);
  fprintf(fp,"Max_Key %" PRIu64 "\t", MAX_KEY_VAL);
  fprintf(fp,"Num_Iters %" PRIu64 "\t", NUM_ITERATIONS);

  switch(SCALING_OPTION){
    case STRONG: {
        fprintf(fp,"Strong Scaling: %" PRIu64 " total keys\t", NUM_KEYS_PER_PE * NUM_PES);
        break;
      }
    case WEAK: {
        fprintf(fp,"Weak Scaling: %" PRIu64 " keys per PE\t", NUM_KEYS_PER_PE);
        break;
      }
    case WEAK_ISOBUCKET: {
        fprintf(fp,"Weak Scaling Constant Bucket Width: %" PRIu64 "u keys per PE \t", NUM_KEYS_PER_PE);
        fprintf(fp,"Constant Bucket Width: %" PRIu64 "\t", BUCKET_WIDTH);
        break;
      }
    default:
      {
        fprintf(fp,"Invalid Scaling Option!\t");
        break;
      }

  }

#ifdef PERMUTE
    fprintf(fp,"Randomized All2All\t");
#elif INCAST
    fprintf(fp,"Incast All2All\t");
#else
    fprintf(fp,"Round Robin All2All\t");
#endif

    fprintf(fp,"\n");
}

/*
 * Prints all of the timining information for an individual PE as a row
 * to the file specificed by 'fp'.
 */
static void print_timer_values(FILE * fp)
{
  for(uint64_t i = 0; i < NUM_ITERATIONS; ++i) {
    for(int t = 0; t < TIMER_NTIMERS; ++t){
      if(timers[t].seconds_iter > 0){
        fprintf(fp,"%f\t", timers[t].seconds[i]);
      }
      if(timers[t].count_iter > 0){
        fprintf(fp,"%u\t", timers[t].count[i]);
      }
    }
    fprintf(fp,"\n");
  }
}

/*
 * Aggregates the per PE timing information
 */
static double * gather_rank_times(_timer_t * const timer)
{
  if(timer->seconds_iter > 0) {

    assert(timer->seconds_iter == timer->num_iters);

    shmem_barrier_all();

#ifdef OPENSHMEM_COMPLIANT
    double * my_average_time = shmem_malloc(sizeof(double));
#else
    double * my_average_time = shmalloc(sizeof(double));
#endif
    double temp = 0.0;
    for(unsigned int i = 0; i < timer->seconds_iter; i++) {
      temp += timer->seconds[i];
    }

    *my_average_time = temp/(timer->seconds_iter);

    double * pe_average_times = shmem_malloc(NUM_PES * sizeof(double));

    shmem_barrier_all();
    shmem_double_fcollect(SHMEM_TEAM_WORLD, pe_average_times, my_average_time, 1);
    shmem_barrier_all();

    shmem_free(my_average_time);

    return pe_average_times;
  }

  else{
    return NULL;
  }
}

/*
 * Aggregates the per PE timing 'count' information
 */
static unsigned int * gather_rank_counts(_timer_t * const timer)
{
  if(timer->count_iter > 0) {

    shmem_barrier_all();

#ifdef OPENSHMEM_COMPLIANT
    unsigned int * my_average_count = shmem_malloc(sizeof(unsigned int));
#else
    unsigned int * my_average_count = shmalloc(sizeof(unsigned int));
#endif
    unsigned int temp = 0;
    for(unsigned int i = 0; i < timer->count_iter; i++) {
      temp += timer->count[i];
    }

    *my_average_count = temp/(timer->count_iter);

    unsigned int * pe_average_counts = shmem_malloc(NUM_PES * sizeof(unsigned int));

    shmem_barrier_all();
    shmem_uint_fcollect(SHMEM_TEAM_WORLD, pe_average_counts, my_average_count, 1);
    shmem_barrier_all();

    shmem_free(my_average_count);

    return pe_average_counts;
  }

  else{
    return NULL;
  }
}

/*
 * Seeds each rank based on the rank number and time
 */
static inline pcg32_random_t seed_my_rank(void)
{
  const unsigned int my_rank = shmem_my_pe();
  pcg32_random_t rng;
  pcg32_srandom_r(&rng, (uint64_t) my_rank, (uint64_t) my_rank );
  return rng;
}

/*
 * Initializes the work array required for SHMEM collective functions
 */
static void init_shmem_sync_array(long * restrict const pSync)
{
  for(uint64_t i = 0; i < _SHMEM_REDUCE_SYNC_SIZE; ++i){
    pSync[i] = _SHMEM_SYNC_VALUE;
  }
  shmem_barrier_all();
}

/*
 * Tests whether or not a file exists.
 * Returns 1 if file exists
 * Returns 0 if file does not exist
 */
static int file_exists(char * filename)
{
  struct stat buffer;

  if(stat(filename,&buffer) == 0){
    return 1;
  }
  else {
    return 0;
  }

}

#ifdef DEBUG
static void wait_my_turn()
{
  shmem_barrier_all();
  whose_turn = 0;
  shmem_barrier_all();
  const int my_rank = shmem_my_pe();

  shmem_int_wait_until((int*)&whose_turn, SHMEM_CMP_EQ, my_rank);
  sleep(1);

}

static void my_turn_complete()
{
  const int my_rank = shmem_my_pe();
  const int next_rank = my_rank+1;

  if(my_rank < (NUM_PES-1)){ // Last rank updates no one
    shmem_int_put((int *) &whose_turn, &next_rank, 1, next_rank);
  }
  shmem_barrier_all();
}
#endif

#ifdef PERMUTE
/*
 * Creates a randomly ordered array of PEs used in the exchange_keys function
 */
static void create_permutation_array()
{

  permute_array = (int *) malloc( NUM_PES * sizeof(int) );

  for(uint64_t i = 0; i < NUM_PES; ++i){
    permute_array[i] = i;
  }

  shuffle(permute_array, NUM_PES);
}

/*
 * Randomly shuffles an array of PE ids
 */
static void shuffle(int* array, size_t n) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    int usec = tv.tv_usec;
    srand48(usec);

    if (n > 1) {
        size_t i;
        for (i = n - 1; i > 0; i--) {
            size_t j = (unsigned int) (drand48()*(i+1));
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}
#endif

