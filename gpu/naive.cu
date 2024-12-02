#include "naive.cuh"

#include <cuda_runtime.h>
#include "nvcomp/lz4.h"

// Error checking helper for CUDA calls
#define CHECK_CUDA(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            return CUDA_ERROR;                                               \
        }                                                                    \
    } while (0)

// Helper function to initialize empty CompressedData
CompressedData *create_compressed_data(size_t num_pages)
{
    CompressedData *data = (CompressedData *)malloc(sizeof(CompressedData));
    if (!data)
        return NULL;

    data->compressed_pages = (CompressedPage *)malloc(num_pages * sizeof(CompressedPage));
    if (!data->compressed_pages)
    {
        free(data);
        return NULL;
    }

    data->num_pages = num_pages;
    data->original_size = 0;

    // Initialize all chunks to NULL
    for (size_t i = 0; i < num_pages; i++)
    {
        data->compressed_pages[i].data = NULL;
        data->compressed_pages[i].size = 0;
    }

    return data;
}

// New helper function to create CompressedData from existing arrays
CompressedData *create_compressed_data_from_arrays(
    const char **page_data_array,   // Array of pointers to compressed page data
    const size_t *page_sizes_array, // Array of compressed page sizes
    size_t num_pages,               // Number of pages
    size_t original_size            // Original uncompressed data size
)
{
    CompressedData *data = (CompressedData *)malloc(sizeof(CompressedData));
    if (!data)
        return NULL;

    data->compressed_pages = (CompressedPage *)malloc(num_pages * sizeof(CompressedPage));
    if (!data->compressed_pages)
    {
        free(data);
        return NULL;
    }

    data->num_pages = num_pages;
    data->original_size = original_size;

    // Copy each page's data
    for (size_t i = 0; i < num_pages; i++)
    {
        data->compressed_pages[i].size = page_sizes_array[i];
        data->compressed_pages[i].data = (char *)malloc(page_sizes_array[i]);

        if (!data->compressed_pages[i].data)
        {
            // Cleanup on failure
            for (size_t j = 0; j < i; j++)
            {
                free(data->compressed_pages[j].data);
            }
            free(data->compressed_pages);
            free(data);
            return NULL;
        }
        memcpy(data->compressed_pages[i].data, page_data_array[i], page_sizes_array[i]);
    }

    return data;
}

// Create CompressedData that references existing arrays without copying
CompressedData *create_compressed_data_with_references(
    const char **page_data_array,   // Array of pointers to compressed page data
    const size_t *page_sizes_array, // Array of compressed page sizes
    size_t num_pages,               // Number of pages
    size_t original_size            // Original uncompressed data size
)
{
    CompressedData *data = (CompressedData *)malloc(sizeof(CompressedData));
    if (!data)
        return NULL;

    data->compressed_pages = (CompressedPage *)malloc(num_pages * sizeof(CompressedPage));
    if (!data->compressed_pages)
    {
        free(data);
        return NULL;
    }

    data->num_pages = num_pages;
    data->original_size = original_size;

    // Store references to each page's data instead of copying
    for (size_t i = 0; i < num_pages; i++)
    {
        data->compressed_pages[i].size = page_sizes_array[i];
        // Simply store the pointer to the original data
        data->compressed_pages[i].data = (char *)page_data_array[i];
    }

    return data;
}

// Helper function to free CompressedData
void free_compressed_data(CompressedData *data)
{
    if (!data)
        return;

    if (data->compressed_pages)
    {
        for (size_t i = 0; i < data->num_pages; i++)
        {
            free(data->compressed_pages[i].data);
        }
        free(data->compressed_pages);
    }

    free(data);
}

void cuda_initialize() {
  cudaFree(0);
}

ErrorCode compress(const char *input_data, size_t in_bytes, CompressedData **output)
{
    // Create timing events
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("\n=== Performance Profile ===\n");
    
    // Phase 1: Initialization
    cudaEventRecord(start);
    
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    const size_t num_pages = (in_bytes + PAGE_SIZE - 1) / PAGE_SIZE;
    // Create output structure
    CompressedData *compressed_result = create_compressed_data(num_pages);
    if (!compressed_result)
    {
        cudaStreamDestroy(stream);
        return MEMORY_ERROR;
    }
    compressed_result->original_size = in_bytes;

    // Allocate device input data
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("1. Initialization: %.3f ms\n", milliseconds);

    // Phase 2: Device Memory Allocation & Initial Transfer
    cudaEventRecord(start);
    
    char *device_input_data;
    if (cudaMalloc(&device_input_data, in_bytes) != cudaSuccess)
    {
        free_compressed_data(compressed_result);
        cudaStreamDestroy(stream);
        return CUDA_ERROR;
    }

    // Copy input data to device
    CHECK_CUDA(cudaMemcpyAsync(device_input_data, input_data, in_bytes,
                               cudaMemcpyHostToDevice, stream));
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("2. Initial Device Allocation & Transfer: %.3f ms\n", milliseconds);

    // Phase 3: Page Management Setup
    cudaEventRecord(start);
    
    // Find bytes per page on input data
    size_t *host_uncompressed_numbytes_per_page;
    CHECK_CUDA(cudaMallocHost(&host_uncompressed_numbytes_per_page, sizeof(size_t) * num_pages));
    for (size_t i = 0; i < num_pages; ++i)
    {
        if (i + 1 < num_pages)
        {
            host_uncompressed_numbytes_per_page[i] = PAGE_SIZE;
        }
        else
        {
            host_uncompressed_numbytes_per_page[i] = in_bytes - (PAGE_SIZE * i);
        }
    }

    // Fill in the pointers to the input data for each page
    void **host_uncompressed_data_per_page;
    CHECK_CUDA(cudaMallocHost(&host_uncompressed_data_per_page, sizeof(void *) * num_pages));
    for (size_t i = 0; i < num_pages; ++i)
    {
        host_uncompressed_data_per_page[i] = device_input_data + PAGE_SIZE * i;
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("3. Page Management Setup: %.3f ms\n", milliseconds);

    // Phase 4: Device Memory Setup for Pages
    cudaEventRecord(start);
    
    // Copy pointers of sizes and data to device
    size_t *device_uncompressed_numbytes_per_page;
    void **device_uncompressed_data_per_page;
    CHECK_CUDA(cudaMalloc(&device_uncompressed_numbytes_per_page, sizeof(size_t) * num_pages));
    CHECK_CUDA(cudaMalloc(&device_uncompressed_data_per_page, sizeof(void *) * num_pages));
    CHECK_CUDA(cudaMemcpyAsync(device_uncompressed_numbytes_per_page, host_uncompressed_numbytes_per_page,
                               sizeof(size_t) * num_pages, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(device_uncompressed_data_per_page, host_uncompressed_data_per_page,
                               sizeof(void *) * num_pages, cudaMemcpyHostToDevice, stream));
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("4. Device Page Setup: %.3f ms\n", milliseconds);


    // Phase 5: Compression Buffer Setup - OPTIMIZED
    cudaEventRecord(start);
    
    size_t temp_bytes;
    nvcompBatchedLZ4CompressGetTempSize(num_pages, PAGE_SIZE,
                                       nvcompBatchedLZ4DefaultOpts, &temp_bytes);
    void *device_temp_ptr;
    CHECK_CUDA(cudaMalloc(&device_temp_ptr, temp_bytes));

    // Allocate a single large buffer instead of many small ones
    void **host_compressed_data_per_page;
    void **device_compressed_data_per_page;
    size_t *device_compressed_numbytes_per_page;
    CHECK_CUDA(cudaMallocHost(&host_compressed_data_per_page, sizeof(void *) * num_pages));
    CHECK_CUDA(cudaMalloc(&device_compressed_data_per_page, sizeof(void *) * num_pages));
    CHECK_CUDA(cudaMalloc(&device_compressed_numbytes_per_page, sizeof(size_t) * num_pages));

    size_t max_out_bytes;
    nvcompBatchedLZ4CompressGetMaxOutputChunkSize(PAGE_SIZE,
                                                 nvcompBatchedLZ4DefaultOpts,
                                                 &max_out_bytes);
    
    // Allocate one large buffer for all pages
    void* compressed_data_buffer;
    CHECK_CUDA(cudaMalloc(&compressed_data_buffer, max_out_bytes * num_pages));
    
    // Set up pointers into the buffer
    for (size_t i = 0; i < num_pages; ++i) {
        host_compressed_data_per_page[i] = (char*)compressed_data_buffer + (i * max_out_bytes);
    }
    CHECK_CUDA(cudaMemcpyAsync(device_compressed_data_per_page, host_compressed_data_per_page,
                              sizeof(void *) * num_pages, cudaMemcpyHostToDevice, stream));
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("5. Compression Buffer Setup: %.3f ms\n", milliseconds);

    // Phase 6: Main Compression
    cudaEventRecord(start);

    // Compress the data
    nvcompStatus_t comp_res = nvcompBatchedLZ4CompressAsync(
        device_uncompressed_data_per_page,
        device_uncompressed_numbytes_per_page,
        PAGE_SIZE,
        num_pages,
        device_temp_ptr,
        temp_bytes,
        device_compressed_data_per_page,
        device_compressed_numbytes_per_page,
        nvcompBatchedLZ4DefaultOpts,
        stream);

    if (comp_res != nvcompSuccess)
    {
        cudaFree(device_input_data);
        cudaFreeHost(host_uncompressed_numbytes_per_page);
        cudaFreeHost(host_uncompressed_data_per_page);
        cudaFree(device_uncompressed_numbytes_per_page);
        cudaFree(device_uncompressed_data_per_page);
        cudaFree(device_temp_ptr);
        for (size_t i = 0; i < num_pages; i++)
        {
            cudaFree(host_compressed_data_per_page[i]);
        }
        cudaFreeHost(host_compressed_data_per_page);
        cudaFree(device_compressed_data_per_page);
        cudaFree(device_compressed_numbytes_per_page);
        free_compressed_data(compressed_result);
        cudaStreamDestroy(stream);
        return NVCOMP_ERROR;
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("6. Main Compression: %.3f ms\n", milliseconds);

    // Phase 7: Result Collection - OPTIMIZED
    cudaEventRecord(start);
    
    // Allocate a single host buffer for all compressed data
    char* host_compressed_buffer;
    CHECK_CUDA(cudaMallocHost(&host_compressed_buffer, max_out_bytes * num_pages));
    
    // Get all compressed sizes in one transfer
    size_t *compressed_sizes = (size_t *)malloc(num_pages * sizeof(size_t));
    if (!compressed_sizes) {
        return MEMORY_ERROR;
    }

    CHECK_CUDA(cudaMemcpyAsync(compressed_sizes, device_compressed_numbytes_per_page,
                              sizeof(size_t) * num_pages, cudaMemcpyDeviceToHost, stream));
    
    // Copy all compressed data in one large transfer
    CHECK_CUDA(cudaMemcpyAsync(host_compressed_buffer, compressed_data_buffer,
                              max_out_bytes * num_pages, cudaMemcpyDeviceToHost, stream));
    
    // Wait for transfers to complete
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    // Set up the output structure (now using CPU memory)
    for (size_t i = 0; i < num_pages; i++) {
        compressed_result->compressed_pages[i].size = compressed_sizes[i];
        compressed_result->compressed_pages[i].data = (char *)malloc(compressed_sizes[i]);
        if (!compressed_result->compressed_pages[i].data) {
            free(compressed_sizes);
            cudaFreeHost(host_compressed_buffer);
            return MEMORY_ERROR;
        }
        
        // Copy from pinned buffer to final destination (CPU memory copy, no CUDA involved) TODO: discuss with nolan, we can copy directly to the right region
        memcpy(compressed_result->compressed_pages[i].data,
               host_compressed_buffer + (i * max_out_bytes),
               compressed_sizes[i]);
    }

    free(compressed_sizes);
    cudaFreeHost(host_compressed_buffer);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("7. Result Collection: %.3f ms\n", milliseconds);

    // Phase 8: Cleanup - OPTIMIZED
    cudaEventRecord(start);
    
    // Single large free instead of many small ones
    cudaFree(compressed_data_buffer);
    cudaFree(device_input_data);
    cudaFreeHost(host_uncompressed_numbytes_per_page);
    cudaFreeHost(host_uncompressed_data_per_page);
    cudaFree(device_uncompressed_numbytes_per_page);
    cudaFree(device_uncompressed_data_per_page);
    cudaFree(device_temp_ptr);
    cudaFreeHost(host_compressed_data_per_page);
    cudaFree(device_compressed_data_per_page);
    cudaFree(device_compressed_numbytes_per_page);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("8. Cleanup: %.3f ms\n", milliseconds);

    // Cleanup timing events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaStreamDestroy(stream);

    *output = compressed_result;
    return SUCCESS;
}

ErrorCode decompress(const CompressedData *compressed_data, char **output_data, size_t *output_size) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    printf("\n=== Starting Unified Memory Version ===\n");
    cudaEventRecord(start);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Get current device for memory advice
    int current_device;
    cudaGetDevice(&current_device);

    size_t num_pages = compressed_data->num_pages;
    *output_size = compressed_data->original_size;

    // Calculate total compressed size
    size_t total_compressed_size = 0;
    size_t *compressed_sizes = (size_t *)malloc(num_pages * sizeof(size_t));
    if (!compressed_sizes) return MEMORY_ERROR;

    for (size_t i = 0; i < num_pages; i++) {
        compressed_sizes[i] = compressed_data->compressed_pages[i].size;
        total_compressed_size += compressed_sizes[i];
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Initial setup: %.3f ms\n", milliseconds);
    cudaEventRecord(start);

    // Allocate unified memory for output early
    CHECK_CUDA(cudaMallocManaged(output_data, compressed_data->original_size));

    // Add memory advice after allocation
    CHECK_CUDA(cudaMemAdvise(*output_data, 
                            compressed_data->original_size,
                            cudaMemAdviseSetPreferredLocation, 
                            current_device));

    // Allocate device buffers
    char *device_compressed_data;
    CHECK_CUDA(cudaMalloc(&device_compressed_data, total_compressed_size));
    void **device_compressed_ptrs;
    CHECK_CUDA(cudaMalloc(&device_compressed_ptrs, num_pages * sizeof(void*)));

    void **host_compressed_ptrs = (void**)malloc(num_pages * sizeof(void*));
    if (!host_compressed_ptrs) {
        cudaFree(device_compressed_data);
        free(compressed_sizes);
        return MEMORY_ERROR;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Initial allocations: %.3f ms\n", milliseconds);
    cudaEventRecord(start);

    // Copy compressed data to GPU
    char* current_pos = device_compressed_data;
    for (size_t i = 0; i < num_pages; i++) {
        host_compressed_ptrs[i] = current_pos;
        CHECK_CUDA(cudaMemcpyAsync(current_pos,
                                compressed_data->compressed_pages[i].data,
                                compressed_sizes[i],
                                cudaMemcpyHostToDevice,
                                stream));
        current_pos += compressed_sizes[i];
    }

    CHECK_CUDA(cudaMemcpyAsync(device_compressed_ptrs, 
                            host_compressed_ptrs,
                            num_pages * sizeof(void*),
                            cudaMemcpyHostToDevice,
                            stream));

    size_t *device_compressed_sizes;
    CHECK_CUDA(cudaMalloc(&device_compressed_sizes, num_pages * sizeof(size_t)));
    CHECK_CUDA(cudaMemcpyAsync(device_compressed_sizes,
                            compressed_sizes,
                            num_pages * sizeof(size_t),
                            cudaMemcpyHostToDevice,
                            stream));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Data transfer to GPU: %.3f ms\n", milliseconds);
    cudaEventRecord(start);

    // Get decompressed sizes
    size_t *device_uncompressed_sizes;
    CHECK_CUDA(cudaMalloc(&device_uncompressed_sizes, num_pages * sizeof(size_t)));

    nvcompStatus_t status = nvcompBatchedLZ4GetDecompressSizeAsync(
        (const void **)device_compressed_ptrs,
        device_compressed_sizes,
        device_uncompressed_sizes,
        num_pages,
        stream);

    if (status != nvcompSuccess) {
        cudaFree(device_compressed_data);
        cudaFree(device_compressed_ptrs);
        cudaFree(device_compressed_sizes);
        cudaFree(device_uncompressed_sizes);
        free(compressed_sizes);
        free(host_compressed_ptrs);
        return NVCOMP_ERROR;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Get decompressed sizes: %.3f ms\n", milliseconds);
    cudaEventRecord(start);

    // Setup output pointers using unified memory
    void **device_uncompressed_ptrs;
    CHECK_CUDA(cudaMalloc(&device_uncompressed_ptrs, num_pages * sizeof(void*)));

    void **host_uncompressed_ptrs = (void**)malloc(num_pages * sizeof(void*));
    if (!host_uncompressed_ptrs) {
        cudaFree(device_compressed_data);
        cudaFree(device_compressed_ptrs);
        cudaFree(device_compressed_sizes);
        cudaFree(device_uncompressed_sizes);
        cudaFree(device_uncompressed_ptrs);
        free(compressed_sizes);
        free(host_compressed_ptrs);
        return MEMORY_ERROR;
    }

    for (size_t i = 0; i < num_pages; i++) {
        host_uncompressed_ptrs[i] = *output_data + (i * PAGE_SIZE);
    }

    CHECK_CUDA(cudaMemcpyAsync(device_uncompressed_ptrs,
                            host_uncompressed_ptrs,
                            num_pages * sizeof(void*),
                            cudaMemcpyHostToDevice,
                            stream));

    size_t temp_bytes;
    nvcompBatchedLZ4DecompressGetTempSize(num_pages, PAGE_SIZE, &temp_bytes);
    void *device_temp;
    CHECK_CUDA(cudaMalloc(&device_temp, temp_bytes));

    nvcompStatus_t *device_statuses;
    CHECK_CUDA(cudaMalloc(&device_statuses, num_pages * sizeof(nvcompStatus_t)));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Output setup: %.3f ms\n", milliseconds);
    cudaEventRecord(start);

    status = nvcompBatchedLZ4DecompressAsync(
        (const void **)device_compressed_ptrs,
        device_compressed_sizes,
        device_uncompressed_sizes,
        device_uncompressed_sizes,
        num_pages,
        device_temp,
        temp_bytes,
        device_uncompressed_ptrs,
        device_statuses,
        stream);

    if (status != nvcompSuccess) {
        cudaFree(device_compressed_data);
        cudaFree(device_compressed_ptrs);
        cudaFree(device_compressed_sizes);
        cudaFree(device_uncompressed_sizes);
        cudaFree(device_uncompressed_ptrs);
        cudaFree(device_temp);
        cudaFree(device_statuses);
        free(compressed_sizes);
        free(host_compressed_ptrs);
        free(host_uncompressed_ptrs);
        return NVCOMP_ERROR;
    }

    cudaStreamSynchronize(stream);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Decompression: %.3f ms\n", milliseconds);
    cudaEventRecord(start);

    // Change preferred location to CPU before prefetching
    CHECK_CUDA(cudaMemAdvise(*output_data,
                            compressed_data->original_size,
                            cudaMemAdviseSetPreferredLocation,
                            cudaCpuDeviceId));

    // Ensure data is available on host
    CHECK_CUDA(cudaMemPrefetchAsync(*output_data, 
                                compressed_data->original_size,
                                cudaCpuDeviceId,
                                stream));

    cudaStreamSynchronize(stream);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Memory prefetch to host: %.3f ms\n", milliseconds);
    cudaEventRecord(start);

    // Cleanup
    cudaFree(device_compressed_data);
    cudaFree(device_compressed_ptrs);
    cudaFree(device_compressed_sizes);
    cudaFree(device_uncompressed_sizes);
    cudaFree(device_uncompressed_ptrs);
    cudaFree(device_temp);
    cudaFree(device_statuses);
    free(compressed_sizes);
    free(host_compressed_ptrs);
    free(host_uncompressed_ptrs);
    cudaStreamDestroy(stream);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Cleanup: %.3f ms\n", milliseconds);
    printf("=== Profiling Complete ===\n\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return SUCCESS;
}

// Helper function to print data
void print_data(const char *data, size_t size, const char *label)
{
    printf("\n%s (first 100 bytes):\n", label);
    for (size_t i = 0; i < (size < 100 ? size : 100); ++i)
    {
        printf("%02x ", (unsigned char)data[i]);
        if ((i + 1) % 20 == 0)
            printf("\n");
    }
    printf("\n");
}

