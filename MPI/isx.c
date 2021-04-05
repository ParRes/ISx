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

#include <mpi.h>
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

uint64_t NUM_PES; // Number of parallel workers
uint64_t TOTAL_KEYS; // Total number of keys across all PEs
uint64_t NUM_KEYS_PER_PE; // Number of keys generated on each PE
uint64_t NUM_BUCKETS; // The number of buckets in the bucket sort
uint64_t BUCKET_WIDTH; // The size of each bucket
uint64_t MAX_KEY_VAL; // The maximum possible generated key value
uint64_t NUM_ITERATIONS; // Number of iterations that the sort is performed

int my_rank;
int comm_size;


#ifdef PERMUTE
int * permute_array;
#endif

int main(int argc,  char ** argv)
{
  //start_pes(0);
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank );
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  char * log_file = parse_params(argc, argv);

  int err =  bucket_sort();

  log_times(log_file);

  MPI_Finalize();
  return err;
}


// Parses all of the command line input and definitions in params.h
// to set all necessary runtime values and options
static char * parse_params(const int argc, char ** argv)
{
  NUM_PES = (uint64_t) comm_size;
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
    if( my_rank == 0){
      printf("Usage:  \n");
      printf("  ./%s <total num keys(strong) | keys per pe(weak)> [iterations] "
             "<log_file>\n",argv[0]);
    }
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
        if(my_rank == 0){
          printf("Invalid scaling option! See params.h to define the scaling option.\n");
        }
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

  if(my_rank == 0){
    printf("ISx MPI 2 sided v%1d.%1d\n",MAJOR_VERSION_NUMBER,MINOR_VERSION_NUMBER);
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

  for(unsigned int i = 0; i < (NUM_ITERATIONS + BURN_IN); ++i)
  {

    // Reset timers after burn in
    if(i == BURN_IN){ init_timers(NUM_ITERATIONS); }

    MPI_Barrier(MPI_COMM_WORLD);

    timer_start(&timers[TIMER_TOTAL]);

    KEY_TYPE * my_keys = make_input();

    int * local_bucket_sizes = count_local_bucket_sizes(my_keys);

    int * send_offsets;
    int * local_bucket_offsets = compute_local_bucket_offsets(local_bucket_sizes,
                                                                   &send_offsets);

    KEY_TYPE * my_local_bucketed_keys =  bucketize_local_keys(my_keys, local_bucket_offsets);

    int * my_global_recv_counts = exchange_receive_counts(local_bucket_sizes);

    int * my_global_recv_offsets = compute_receive_offsets(my_global_recv_counts);

    long long int  my_bucket_size;
    KEY_TYPE * my_bucket_keys = exchange_keys(my_global_recv_offsets,
                                              my_global_recv_counts,
                                              send_offsets,
                                              local_bucket_sizes,
                                              my_local_bucketed_keys,
                                              &my_bucket_size);


    int * my_local_key_counts = count_local_keys(my_bucket_keys, my_bucket_size);

    MPI_Barrier(MPI_COMM_WORLD);

    timer_stop(&timers[TIMER_TOTAL]);

    // Only the last iteration is verified
    if(i == NUM_ITERATIONS) {
      err = verify_results(my_local_key_counts, my_bucket_keys, my_bucket_size);
    }


    free(my_local_bucketed_keys);
    free(my_keys);
    free(local_bucket_sizes);
    free(local_bucket_offsets);
    free(send_offsets);
    free(my_local_key_counts);
    free(my_bucket_keys);

    MPI_Barrier(MPI_COMM_WORLD);
  }
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

  for(unsigned int i = 0; i < NUM_KEYS_PER_PE; ++i) {
    my_keys[i] = pcg32_boundedrand_r(&rng, MAX_KEY_VAL);
  }

  timer_stop(&timers[TIMER_INPUT]);

#ifdef DEBUG

  char msg[1024];
  sprintf(msg,"Rank %d: Initial Keys: ", my_rank);
  for(int i = 0; i < NUM_KEYS_PER_PE; ++i){
    if(i < PRINT_MAX)
    sprintf(msg + strlen(msg),"%d ", my_keys[i]);
  }
  sprintf(msg + strlen(msg),"\n");
  printf("%s",msg);
  fflush(stdout);

#endif
  return my_keys;
}


/*
 * Computes the size of each bucket by iterating all keys and incrementing
 * their corresponding bucket's size
 */
static inline int * count_local_bucket_sizes(KEY_TYPE const * restrict const my_keys)
{
  int * restrict const local_bucket_sizes =  malloc(NUM_BUCKETS * sizeof(int));

  timer_start(&timers[TIMER_BCOUNT]);

  init_array(local_bucket_sizes, NUM_BUCKETS);

  for(unsigned int i = 0; i < NUM_KEYS_PER_PE; ++i){
    const uint32_t bucket_index = my_keys[i]/BUCKET_WIDTH;
    local_bucket_sizes[bucket_index]++;
  }

  timer_stop(&timers[TIMER_BCOUNT]);

#ifdef DEBUG

  char msg[1024];
  sprintf(msg,"Rank %d: local bucket sizes: ", my_rank);
  for(int i = 0; i < NUM_BUCKETS; ++i){
    if(i < PRINT_MAX)
    sprintf(msg + strlen(msg),"%d ", local_bucket_sizes[i]);
  }
  sprintf(msg + strlen(msg),"\n");
  printf("%s",msg);
  fflush(stdout);

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
  int * restrict const local_bucket_offsets =  malloc(NUM_BUCKETS * sizeof(int));

  timer_start(&timers[TIMER_BOFFSET]);

  (*send_offsets) =  malloc(NUM_BUCKETS * sizeof(int));

  local_bucket_offsets[0] = 0;
  (*send_offsets)[0] = 0;
  int temp = 0;
  for(unsigned int i = 1; i < NUM_BUCKETS; i++){
    temp = local_bucket_offsets[i-1] + local_bucket_sizes[i-1];
    local_bucket_offsets[i] = temp;
    (*send_offsets)[i] = temp;
  }
  timer_stop(&timers[TIMER_BOFFSET]);

#ifdef DEBUG

  char msg[1024];
  sprintf(msg,"Rank %d: local bucket offsets: ", my_rank);
  for(int i = 0; i < NUM_BUCKETS; ++i){
    if(i < PRINT_MAX)
    sprintf(msg + strlen(msg),"%d ", local_bucket_offsets[i]);
  }
  sprintf(msg + strlen(msg),"\n");
  printf("%s",msg);
  fflush(stdout);

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

  for(unsigned int i = 0; i < NUM_KEYS_PER_PE; ++i){
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

  char msg[1024];
  sprintf(msg,"Rank %d: local bucketed keys: ", my_rank);
  for(int i = 0; i < NUM_KEYS_PER_PE; ++i){
    if(i < PRINT_MAX)
    sprintf(msg + strlen(msg),"%d ", my_local_bucketed_keys[i]);
  }
  sprintf(msg + strlen(msg),"\n");
  printf("%s",msg);
  fflush(stdout);

#endif
  return my_local_bucketed_keys;
}


/*
 * Each PE tells every other PE how many elements will be sent to them
 */
static int * exchange_receive_counts(int const * restrict const local_bucket_sizes)
{

  int * restrict const my_global_recv_counts = malloc(NUM_PES * sizeof(int));

  timer_start(&timers[TIMER_ATA_COUNTS]);

  MPI_Alltoall( local_bucket_sizes,
                1, MPI_INT,
                my_global_recv_counts,
                1, MPI_INT,
                MPI_COMM_WORLD);

  timer_stop(&timers[TIMER_ATA_COUNTS]);

#ifdef DEBUG
  MPI_Barrier(MPI_COMM_WORLD);
  char msg[1024];
  sprintf(msg,"Rank %d: global receive counts: ", my_rank);
  for(int i = 0; i < NUM_BUCKETS; ++i){
    sprintf(msg + strlen(msg),"%d ", my_global_recv_counts[i]);
  }
  sprintf(msg + strlen(msg),"\n");
  printf("%s",msg);
  fflush(stdout);
#endif

  return my_global_recv_counts;
}

/*
 * Computes a prefix sum of the global receive counts to determine the write
 * offset for remote PEs into the local receiver array
 */
static int * compute_receive_offsets(int const * restrict const my_global_recv_counts)
{
  // +1 to store the total number of keys to be received
  // so you know how large to make your receive array.
  // Last element is the total number of keys to receive.
  const int receive_offsets_size = NUM_PES + 1;

  timer_start(&timers[TIMER_RECV_OFFSET]);
  int * restrict const my_global_recv_offsets = malloc(receive_offsets_size * sizeof(int));

  my_global_recv_offsets[0] = 0;

  for(int i = 1; i < receive_offsets_size; ++i){
    my_global_recv_offsets[i] = my_global_recv_offsets[i-1] + my_global_recv_counts[i-1];
  }

  timer_stop(&timers[TIMER_RECV_OFFSET]);

#ifdef DEBUG
  MPI_Barrier(MPI_COMM_WORLD);
  char msg[1024];
  sprintf(msg,"Rank %d: global receive offsets: ", my_rank);
  for(int i = 0; i < NUM_BUCKETS; ++i){
    sprintf(msg + strlen(msg),"%d ", my_global_recv_offsets[i]);
  }
  sprintf(msg + strlen(msg),"\n");
  printf("%s",msg);
  fflush(stdout);
#endif

  return my_global_recv_offsets;
}

/*
 * Each PE sends the contents of its local buckets to the PE that owns that bucket.
 */
static inline KEY_TYPE * exchange_keys( int const * restrict const global_recv_offsets,
                                        int const * restrict const global_recv_counts,
                                        int const * restrict const send_offsets,
                                        int const * restrict const local_bucket_sizes,
                                        KEY_TYPE const * restrict const my_local_bucketed_keys,
                                        long long int * my_bucket_size)
{
  timer_start(&timers[TIMER_ATA_KEYS]);

  //unsigned int total_keys_sent = 0;
  (*my_bucket_size) = global_recv_offsets[NUM_PES];

  KEY_TYPE * restrict const my_bucket_keys = malloc((*my_bucket_size)*sizeof(KEY_TYPE));

  MPI_Alltoallv(my_local_bucketed_keys, local_bucket_sizes, send_offsets, MPI_INT,
                my_bucket_keys, global_recv_counts, global_recv_offsets, MPI_INT,
                MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  timer_stop(&timers[TIMER_ATA_KEYS]);

#ifdef DEBUG
  MPI_Barrier(MPI_COMM_WORLD);
  char msg[1024];
  sprintf(msg,"Rank %d: Bucket Size %lld | | Keys after exchange:", my_rank, *my_bucket_size);
  for(int i = 0; i < *my_bucket_size; ++i){
    if(i < PRINT_MAX)
    sprintf(msg + strlen(msg),"%d ", my_bucket_keys[i]);
  }
  sprintf(msg + strlen(msg),"\n");
  printf("%s",msg);
  fflush(stdout);
#endif

  return my_bucket_keys;
}


/*
 * Counts the occurence of each key in my bucket.
 * Key indices into the count array are the key's value minus my bucket's
 * minimum key value to allow indexing from 0.
 * my_bucket_keys: All keys in my bucket unsorted [my_rank * BUCKET_WIDTH, (my_rank+1)*BUCKET_WIDTH)
 */
static inline int * count_local_keys(KEY_TYPE const * restrict const my_bucket_keys,
                                          const long long int my_bucket_size)
{
  int * restrict const my_local_key_counts = malloc(BUCKET_WIDTH * sizeof(int));
  memset(my_local_key_counts, 0, BUCKET_WIDTH * sizeof(int));

  timer_start(&timers[TIMER_SORT]);

  const int my_min_key = my_rank * BUCKET_WIDTH;

  // Count the occurences of each key in my bucket
  for(int i = 0; i < my_bucket_size; ++i){
    const unsigned int key_index = my_bucket_keys[i] - my_min_key;

    assert(my_bucket_keys[i] >= my_min_key);
    assert(key_index < BUCKET_WIDTH);

    my_local_key_counts[key_index]++;
  }
  timer_stop(&timers[TIMER_SORT]);

#ifdef DEBUG

  char msg[4096];
  sprintf(msg,"Rank %d: Bucket Size %lld | Local Key Counts:", my_rank, my_bucket_size);
  for(int i = 0; i < BUCKET_WIDTH; ++i){
    if(i < PRINT_MAX)
    sprintf(msg + strlen(msg),"%d ", my_local_key_counts[i]);
  }
  sprintf(msg + strlen(msg),"\n");
  printf("%s",msg);
  fflush(stdout);

#endif

  return my_local_key_counts;
}

/*
 * Verifies the correctness of the sort.
 * Ensures all keys are within a PE's bucket boundaries.
 * Ensures the final number of keys is equal to the initial.
 */
static int verify_results(int const * restrict const my_local_key_counts,
                           KEY_TYPE const * restrict const my_local_keys,
                           const long long int my_bucket_size)
{

  MPI_Barrier(MPI_COMM_WORLD);

  int error = 0;

  const int my_min_key = my_rank * BUCKET_WIDTH;
  const int my_max_key = (my_rank+1) * BUCKET_WIDTH - 1;

  // Verify all keys are within bucket boundaries
  for(int i = 0; i < my_bucket_size; ++i){
    const int key = my_local_keys[i];
    if((key < my_min_key) || (key > my_max_key)){
      printf("Rank %d Failed Verification!\n",my_rank);
      printf("Key: %d is outside of bounds [%d, %d]\n", key, my_min_key, my_max_key);
      error = 1;
    }
  }

  // Verify the sum of the key population equals the expected bucket size
  int bucket_size_test = 0;
  for(unsigned int i = 0; i < BUCKET_WIDTH; ++i){
    bucket_size_test += my_local_key_counts[i];
  }
  if(bucket_size_test != my_bucket_size){
      printf("Rank %d Failed Verification!\n",my_rank);
      printf("Actual Bucket Size: %d Should be %lld\n", bucket_size_test, my_bucket_size);
      error = 1;
  }

  // Verify the final number of keys equals the initial number of keys
  long long int total_num_keys = 0;
  MPI_Allreduce(&my_bucket_size, &total_num_keys, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);

  if(total_num_keys != (long long int)(NUM_KEYS_PER_PE * NUM_PES)){
    if(my_rank == ROOT_PE){
      printf("Verification Failed!\n");
      printf("Actual total number of keys: %lld", total_num_keys );
      printf(" Expected %" PRId64 "\n", NUM_KEYS_PER_PE * NUM_PES );
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

  if(my_rank == ROOT_PE)
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

  for (int i = 0; i < comm_size; ++i) {
    if (i == my_rank) {
      if((fp = fopen(log_file, "a+b"))==NULL){
        perror("Error opening log file:");
        exit(1);
      }

        print_timer_values(fp);
        fclose(fp);
      }
      MPI_Barrier(MPI_COMM_WORLD);
  }

  for(int i = 0; i < TIMER_NTIMERS; ++i){
    timers[i].pe_average_times = gather_rank_times(&timers[i]);
    // No need gather the average counts since we are not currently reporting them
    //timers[i].pe_average_counts = gather_rank_counts(&timers[i]);
  }

  if(my_rank == ROOT_PE)
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
    for(unsigned int i = 0; i < NUM_PES; ++i){
      temp += timers[TIMER_TOTAL].pe_average_times[i];
    }
    printf("Average total time (per PE): %f seconds\n", temp/NUM_PES);
  }

  if(timers[TIMER_ATA_KEYS].seconds_iter >0) {
    double temp = 0.0;
    for(unsigned int i = 0; i < NUM_PES; ++i){
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
  for(int i = 0; i < TIMER_NTIMERS; ++i){
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
  fprintf(fp,"MPI\t");
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
        fprintf(fp,"Weak Scaling Constant Bucket Width: %" PRIu64 " keys per PE \t", NUM_KEYS_PER_PE);
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
  unsigned int num_records = NUM_PES * NUM_ITERATIONS;

  for(unsigned int i = 0; i < num_records; ++i) {
    for(int t = 0; t < TIMER_NTIMERS; ++t){
      if(timers[t].seconds_iter > 0){
        fprintf(fp,"%f\t", timers[t].seconds[t]);
      }
      if(timers[t].count_iter > 0){
        fprintf(fp,"%u\t", timers[t].count[t]);
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

    double my_average_time;
    double * restrict pe_average_times = NULL;
    if (my_rank == ROOT_PE) {
      pe_average_times = malloc(NUM_PES * sizeof(double));
    }

    double temp = 0.0;
    for(unsigned int i = 0; i < timer->seconds_iter; ++i) {
      temp += timer->seconds[i];
    }
    my_average_time = temp/(timer->seconds_iter);

    MPI_Gather(&my_average_time, 1, MPI_DOUBLE,
               pe_average_times, 1, MPI_DOUBLE,
               ROOT_PE, MPI_COMM_WORLD);

    return pe_average_times;
  }
  else{
    return NULL;
  }
}

static unsigned int * gather_rank_counts(_timer_t * const timer)
{
  if(timer->count_iter > 0) {

    unsigned int my_average_count;
    unsigned int * restrict pe_average_counts = NULL;
    if (my_rank == ROOT_PE) {
      pe_average_counts = malloc(NUM_PES * sizeof(unsigned int));
    }

    unsigned int temp = 0;
    for(unsigned int i = 0; i < timer->count_iter; ++i) {
      temp += timer->count[i];
    }
    my_average_count = temp/(timer->count_iter);

    MPI_Gather(&my_average_count, 1, MPI_DOUBLE,
               pe_average_counts, 1, MPI_DOUBLE,
               ROOT_PE, MPI_COMM_WORLD);

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
  pcg32_random_t rng;
  pcg32_srandom_r(&rng, (uint64_t) my_rank, (uint64_t) my_rank );
  return rng;
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

#ifdef PERMUTE
/*
 * Creates a randomly ordered array of PEs used in the exchange_keys function
 */
static void create_permutation_array()
{

  permute_array = malloc( NUM_PES * sizeof(int) );

  for(unsigned int i = 0; i < NUM_PES; ++i){
    permute_array[i] = i;
  }

  shuffle(permute_array, NUM_PES, sizeof(int));
}

/*
 * Randomly shuffles a generic array
 */
static void shuffle(void * array, size_t n, size_t size)
{
  char tmp[size];
  char * arr = array;
  size_t stride = size * sizeof(char);
  if(n > 1){
    for(int i = 0; i < (n - 1); ++i){
      size_t rnd = (size_t) rand();
      size_t j = i + rnd/(RAND_MAX/(n - i) + 1);
      memcpy(tmp, arr + j*stride, size);
      memcpy(arr + j*stride, arr + i*stride, size);
      memcpy(arr + i*stride, tmp, size);
    }
  }
}
#endif

