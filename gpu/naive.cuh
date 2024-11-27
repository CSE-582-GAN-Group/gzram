/*
Some TODOs:
- add better support to directly create CompressedData from compressed pages and sizes
- lots of small optimizations. I can probably use less data structures and less memory allocations.
- cudaMallocManaged()
- Pipeline the compression and decompression steps
- Group data into chunks and compress them one at a time
- More async memcpys
- change to ZSTD
*/
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Page size for compression
#define PAGE_SIZE 4096

#if __cplusplus
extern "C" {
#endif

// Error handling enum
typedef enum
{
    SUCCESS = 0,
    CUDA_ERROR,
    NVCOMP_ERROR,
    MEMORY_ERROR
} ErrorCode;

typedef struct
{
    char *data;
    size_t size;
} CompressedPage;

// Main compression data structure - removed compressed_page_sizes as it's redundant
typedef struct
{
    CompressedPage *compressed_pages; // Array of compressed pages
    size_t num_pages;                 // Number of pages
    size_t original_size;             // Size of the original data
} CompressedData;

// Helper function to initialize empty CompressedData
CompressedData *create_compressed_data(size_t num_pages);

// New helper function to create CompressedData from existing arrays
CompressedData *create_compressed_data_from_arrays(
    const char **page_data_array,   // Array of pointers to compressed page data
    const size_t *page_sizes_array, // Array of compressed page sizes
    size_t num_pages,               // Number of pages
    size_t original_size            // Original uncompressed data size
);

// Create CompressedData that references existing arrays without copying
CompressedData *create_compressed_data_with_references(
    const char **page_data_array,   // Array of pointers to compressed page data
    const size_t *page_sizes_array, // Array of compressed page sizes
    size_t num_pages,               // Number of pages
    size_t original_size            // Original uncompressed data size
);

// Helper function to free CompressedData
void free_compressed_data(CompressedData *data);

ErrorCode compress(const char *input_data, size_t in_bytes, CompressedData **output);

ErrorCode decompress(const CompressedData *compressed_data, char **output_data, size_t *output_size);

// Helper function to print data
void print_data(const char *data, size_t size, const char *label);

#if __cplusplus
}
#endif