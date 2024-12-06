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

// Initialize CUDA
void cuda_initialize()
{
    cudaFree(0);
}

void cuda_free(void *ptr) {
  cudaFree(ptr);
}

ErrorCode compress_pipelined(const char *input_data, size_t in_bytes,
                             CompressedData **output)
{
#ifdef DEBUG
    printf("\n=== Pipelined Compression ===\n");
#endif
    int num_batches = 5;
    int num_streams = 3;

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // printf("\n=== Pipelined Performance Profile ===\n");

    // Phase 1: Initial Setup
    cudaEventRecord(start);
    const size_t num_pages = (in_bytes + PAGE_SIZE - 1) / PAGE_SIZE;
    const size_t pages_per_batch = (num_pages + num_batches - 1) / num_batches;

    CompressedData *compressed_result = create_compressed_data(num_pages);
    if (!compressed_result)
    {
        return MEMORY_ERROR;
    }
    compressed_result->original_size = in_bytes;

    // Create stream array
    cudaStream_t *streams = (cudaStream_t *)malloc(num_streams * sizeof(cudaStream_t));
    for (int i = 0; i < num_streams; i++)
    {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("1. Initial Setup: %.3f ms\n", milliseconds);

    // Phase 2: Resource Allocation
    cudaEventRecord(start);

    // Allocate device resources for each stream
    struct StreamResources
    {
        char *device_input_data;
        size_t *device_uncompressed_numbytes;
        void **device_uncompressed_data;
        void *device_temp_ptr;
        void **device_compressed_data;
        size_t *device_compressed_numbytes;
        void *compressed_data_buffer;
    };

    StreamResources *resources = (StreamResources *)malloc(num_streams * sizeof(StreamResources));

    size_t max_batch_bytes = pages_per_batch * PAGE_SIZE;
    size_t temp_bytes;
    nvcompBatchedLZ4CompressGetTempSize(pages_per_batch, PAGE_SIZE,
                                        nvcompBatchedLZ4DefaultOpts, &temp_bytes);

    size_t max_out_bytes;
    nvcompBatchedLZ4CompressGetMaxOutputChunkSize(PAGE_SIZE,
                                                  nvcompBatchedLZ4DefaultOpts,
                                                  &max_out_bytes);

    // Allocate resources for each stream
    for (int i = 0; i < num_streams; i++)
    {
        CHECK_CUDA(cudaMalloc(&resources[i].device_input_data, max_batch_bytes));
        CHECK_CUDA(cudaMalloc(&resources[i].device_uncompressed_numbytes,
                              sizeof(size_t) * pages_per_batch));
        CHECK_CUDA(cudaMalloc(&resources[i].device_uncompressed_data,
                              sizeof(void *) * pages_per_batch));
        CHECK_CUDA(cudaMalloc(&resources[i].device_temp_ptr, temp_bytes));
        CHECK_CUDA(cudaMalloc(&resources[i].device_compressed_data,
                              sizeof(void *) * pages_per_batch));
        CHECK_CUDA(cudaMalloc(&resources[i].device_compressed_numbytes,
                              sizeof(size_t) * pages_per_batch));
        CHECK_CUDA(cudaMalloc(&resources[i].compressed_data_buffer,
                              max_out_bytes * pages_per_batch));
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("2. Resource Allocation: %.3f ms\n", milliseconds);

    // Phase 3: Pipelined Compression
    cudaEventRecord(start);

    for (size_t batch = 0; batch < num_batches; batch++)
    {
        int stream_idx = batch % num_streams;
        size_t start_page = batch * pages_per_batch;
        size_t batch_pages = min(pages_per_batch, num_pages - start_page);
        size_t batch_bytes = min(max_batch_bytes, in_bytes - (start_page * PAGE_SIZE));

        // Setup batch metadata
        void **host_uncompressed_data = (void **)malloc(batch_pages * sizeof(void *));
        size_t *host_uncompressed_sizes = (size_t *)malloc(batch_pages * sizeof(size_t));
        void **host_compressed_data = (void **)malloc(batch_pages * sizeof(void *));

        for (size_t i = 0; i < batch_pages; i++)
        {
            size_t page_offset = (start_page + i) * PAGE_SIZE;
            host_uncompressed_sizes[i] = min((unsigned long)PAGE_SIZE, in_bytes - page_offset);
            host_uncompressed_data[i] = resources[stream_idx].device_input_data +
                                        (i * PAGE_SIZE);
            host_compressed_data[i] = (char *)resources[stream_idx].compressed_data_buffer +
                                      (i * max_out_bytes);
        }

        // Copy input data and metadata
        CHECK_CUDA(cudaMemcpyAsync(resources[stream_idx].device_input_data,
                                   input_data + (start_page * PAGE_SIZE),
                                   batch_bytes, cudaMemcpyHostToDevice,
                                   streams[stream_idx]));

        CHECK_CUDA(cudaMemcpyAsync(resources[stream_idx].device_uncompressed_numbytes,
                                   host_uncompressed_sizes,
                                   sizeof(size_t) * batch_pages,
                                   cudaMemcpyHostToDevice, streams[stream_idx]));

        CHECK_CUDA(cudaMemcpyAsync(resources[stream_idx].device_uncompressed_data,
                                   host_uncompressed_data,
                                   sizeof(void *) * batch_pages,
                                   cudaMemcpyHostToDevice, streams[stream_idx]));

        CHECK_CUDA(cudaMemcpyAsync(resources[stream_idx].device_compressed_data,
                                   host_compressed_data,
                                   sizeof(void *) * batch_pages,
                                   cudaMemcpyHostToDevice, streams[stream_idx]));

        // Compress batch
        nvcompStatus_t comp_res = nvcompBatchedLZ4CompressAsync(
            resources[stream_idx].device_uncompressed_data,
            resources[stream_idx].device_uncompressed_numbytes,
            PAGE_SIZE, batch_pages,
            resources[stream_idx].device_temp_ptr,
            temp_bytes,
            resources[stream_idx].device_compressed_data,
            resources[stream_idx].device_compressed_numbytes,
            nvcompBatchedLZ4DefaultOpts,
            streams[stream_idx]);

        if (comp_res != nvcompSuccess)
        {
            // Cleanup and return error
            for (int i = 0; i < num_streams; i++)
            {
                cudaStreamDestroy(streams[i]);
                cudaFree(resources[i].device_input_data);
                cudaFree(resources[i].device_uncompressed_numbytes);
                cudaFree(resources[i].device_uncompressed_data);
                cudaFree(resources[i].device_temp_ptr);
                cudaFree(resources[i].device_compressed_data);
                cudaFree(resources[i].device_compressed_numbytes);
                cudaFree(resources[i].compressed_data_buffer);
            }
            free(streams);
            free(resources);
            free_compressed_data(compressed_result);
            return NVCOMP_ERROR;
        }

        // Allocate host buffer for compressed data
        char *host_compressed_buffer;
        CHECK_CUDA(cudaMallocHost(&host_compressed_buffer,
                                  max_out_bytes * batch_pages));

        // Get compressed sizes
        size_t *compressed_sizes = (size_t *)malloc(batch_pages * sizeof(size_t));
        CHECK_CUDA(cudaMemcpyAsync(compressed_sizes,
                                   resources[stream_idx].device_compressed_numbytes,
                                   sizeof(size_t) * batch_pages,
                                   cudaMemcpyDeviceToHost, streams[stream_idx]));

        // Copy compressed data
        CHECK_CUDA(cudaMemcpyAsync(host_compressed_buffer,
                                   resources[stream_idx].compressed_data_buffer,
                                   max_out_bytes * batch_pages,
                                   cudaMemcpyDeviceToHost, streams[stream_idx]));

        // Wait for stream operations to complete
        CHECK_CUDA(cudaStreamSynchronize(streams[stream_idx]));

        // Store results
        for (size_t i = 0; i < batch_pages; i++)
        {
            size_t page_idx = start_page + i;
            compressed_result->compressed_pages[page_idx].size = compressed_sizes[i];
            compressed_result->compressed_pages[page_idx].data =
                (char *)malloc(compressed_sizes[i]);

            if (!compressed_result->compressed_pages[page_idx].data)
            {
                cudaFreeHost(host_compressed_buffer);
                free(compressed_sizes);
                free(host_uncompressed_data);
                free(host_uncompressed_sizes);
                free(host_compressed_data);
                return MEMORY_ERROR;
            }

            memcpy(compressed_result->compressed_pages[page_idx].data,
                   host_compressed_buffer + (i * max_out_bytes),
                   compressed_sizes[i]);
        }

        // Cleanup batch resources
        cudaFreeHost(host_compressed_buffer);
        free(compressed_sizes);
        free(host_uncompressed_data);
        free(host_uncompressed_sizes);
        free(host_compressed_data);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("3. Pipelined Compression: %.3f ms\n", milliseconds);

    // Phase 4: Cleanup
    cudaEventRecord(start);

    for (int i = 0; i < num_streams; i++)
    {
        cudaStreamDestroy(streams[i]);
        cudaFree(resources[i].device_input_data);
        cudaFree(resources[i].device_uncompressed_numbytes);
        cudaFree(resources[i].device_uncompressed_data);
        cudaFree(resources[i].device_temp_ptr);
        cudaFree(resources[i].device_compressed_data);
        cudaFree(resources[i].device_compressed_numbytes);
        cudaFree(resources[i].compressed_data_buffer);
    }

    free(streams);
    free(resources);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("4. Cleanup: %.3f ms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    *output = compressed_result;
    return SUCCESS;
}

ErrorCode compress_improved_naive(const char *input_data, size_t in_bytes, CompressedData **output)
{
//    printf("\n=== Improved Naive Compression ===\n");

    // Create timing events
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // printf("\n=== Performance Profile ===\n");

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
    // printf("1. Initialization: %.3f ms\n", milliseconds);

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
    // printf("2. Initial Device Allocation & Transfer: %.3f ms\n", milliseconds);

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
    // printf("3. Page Management Setup: %.3f ms\n", milliseconds);

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
    // printf("4. Device Page Setup: %.3f ms\n", milliseconds);

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
    void *compressed_data_buffer;
    CHECK_CUDA(cudaMalloc(&compressed_data_buffer, max_out_bytes * num_pages));

    // Set up pointers into the buffer
    for (size_t i = 0; i < num_pages; ++i)
    {
        host_compressed_data_per_page[i] = (char *)compressed_data_buffer + (i * max_out_bytes);
    }
    CHECK_CUDA(cudaMemcpyAsync(device_compressed_data_per_page, host_compressed_data_per_page,
                               sizeof(void *) * num_pages, cudaMemcpyHostToDevice, stream));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("5. Compression Buffer Setup: %.3f ms\n", milliseconds);

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
    // printf("6. Main Compression: %.3f ms\n", milliseconds);

    // Phase 7: Result Collection - OPTIMIZED
    cudaEventRecord(start);

    // Allocate a single host buffer for all compressed data
    char *host_compressed_buffer;
    CHECK_CUDA(cudaMallocHost(&host_compressed_buffer, max_out_bytes * num_pages));

    // Get all compressed sizes in one transfer
    size_t *compressed_sizes = (size_t *)malloc(num_pages * sizeof(size_t));
    if (!compressed_sizes)
    {
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
    for (size_t i = 0; i < num_pages; i++)
    {
        compressed_result->compressed_pages[i].size = compressed_sizes[i];
        compressed_result->compressed_pages[i].data = (char *)malloc(compressed_sizes[i]);
        if (!compressed_result->compressed_pages[i].data)
        {
            free(compressed_sizes);
            cudaFreeHost(host_compressed_buffer);
            return MEMORY_ERROR;
        }

        // Copy from pinned buffer to final destination (CPU memory copy, no CUDA involved)
        memcpy(compressed_result->compressed_pages[i].data,
               host_compressed_buffer + (i * max_out_bytes),
               compressed_sizes[i]);
    }

    free(compressed_sizes);
    cudaFreeHost(host_compressed_buffer);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("7. Result Collection: %.3f ms\n", milliseconds);

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
    // printf("8. Cleanup: %.3f ms\n", milliseconds);

    // Cleanup timing events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaStreamDestroy(stream);

    *output = compressed_result;
    return SUCCESS;
}

ErrorCode compress_naive(const char *input_data, size_t in_bytes, CompressedData **output)
{
//    printf("\n=== Naive Compression ===\n");
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

    // Copy pointers of sizes and data to device
    size_t *device_uncompressed_numbytes_per_page;
    void **device_uncompressed_data_per_page;
    CHECK_CUDA(cudaMalloc(&device_uncompressed_numbytes_per_page, sizeof(size_t) * num_pages));
    CHECK_CUDA(cudaMalloc(&device_uncompressed_data_per_page, sizeof(void *) * num_pages));
    CHECK_CUDA(cudaMemcpyAsync(device_uncompressed_numbytes_per_page, host_uncompressed_numbytes_per_page,
                               sizeof(size_t) * num_pages, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(device_uncompressed_data_per_page, host_uncompressed_data_per_page,
                               sizeof(void *) * num_pages, cudaMemcpyHostToDevice, stream));

    // Calculate temp data buffer size required for compression and allocate it
    size_t temp_bytes;
    nvcompBatchedLZ4CompressGetTempSize(num_pages, PAGE_SIZE,
                                        nvcompBatchedLZ4DefaultOpts, &temp_bytes);
    void *device_temp_ptr;
    CHECK_CUDA(cudaMalloc(&device_temp_ptr, temp_bytes));

    // Allocate space for compressed data
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
    for (size_t i = 0; i < num_pages; ++i)
    {
        CHECK_CUDA(cudaMalloc(&host_compressed_data_per_page[i], max_out_bytes));
    }
    CHECK_CUDA(cudaMemcpyAsync(device_compressed_data_per_page, host_compressed_data_per_page,
                               sizeof(void *) * num_pages, cudaMemcpyHostToDevice, stream));

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

    // Get the compressed sizes and copy compressed data
    size_t *compressed_sizes = (size_t *)malloc(num_pages * sizeof(size_t));
    if (!compressed_sizes)
    {
        // TODO: cleanup resources
        return MEMORY_ERROR;
    }

    CHECK_CUDA(cudaMemcpy(compressed_sizes, device_compressed_numbytes_per_page,
                          sizeof(size_t) * num_pages, cudaMemcpyDeviceToHost));

    // Copy each compressed chunk
    for (size_t i = 0; i < num_pages; i++)
    {
        compressed_result->compressed_pages[i].size = compressed_sizes[i];
        compressed_result->compressed_pages[i].data = (char *)malloc(compressed_sizes[i]);
        if (!compressed_result->compressed_pages[i].data)
        {
            free(compressed_sizes);
            free_compressed_data(compressed_result);
            // TODO: cleanup CUDA resources
            return MEMORY_ERROR;
        }

        CHECK_CUDA(cudaMemcpy(compressed_result->compressed_pages[i].data,
                              host_compressed_data_per_page[i],
                              compressed_sizes[i],
                              cudaMemcpyDeviceToHost));
    }

    free(compressed_sizes);

    // Cleanup CUDA resources
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
    cudaStreamDestroy(stream);

    *output = compressed_result;
    return SUCCESS;
}

ErrorCode decompress_pipelined(const CompressedData *compressed_data, char **output_data, size_t *output_size)
{
    printf("\n=== Pipelined Decompression ===\n");
    const int NUM_STREAMS = 3;  // Number of concurrent streams
    const int NUM_BATCHES = 10; // Number of batches to split the original data into

    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t events[NUM_STREAMS];

    // Create streams and events
    for (int i = 0; i < NUM_STREAMS; i++)
    {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
        CHECK_CUDA(cudaEventCreate(&events[i]));
    }

    size_t num_pages = compressed_data->num_pages;
    *output_size = compressed_data->original_size;

    // Calculate pages per batch
    size_t pages_per_batch = (num_pages + NUM_BATCHES - 1) / NUM_BATCHES;

    // Allocate unified memory for output
    CHECK_CUDA(cudaMallocManaged(output_data, compressed_data->original_size));

    // Process each batch
    for (size_t batch = 0; batch < NUM_BATCHES; batch++)
    {
        size_t batch_start = batch * pages_per_batch;
        size_t batch_end = min(batch_start + pages_per_batch, num_pages);
        size_t batch_pages = batch_end - batch_start;

        if (batch_pages == 0)
            break;

        // Calculate total compressed size for this batch
        size_t batch_compressed_size = 0;
        for (size_t i = batch_start; i < batch_end; i++)
        {
            batch_compressed_size += compressed_data->compressed_pages[i].size;
        }

        // Pages per stream for this batch
        size_t pages_per_stream = (batch_pages + NUM_STREAMS - 1) / NUM_STREAMS;

        // Process the batch using multiple streams
        for (int stream_idx = 0; stream_idx < NUM_STREAMS; stream_idx++)
        {
            size_t stream_start = batch_start + (stream_idx * pages_per_stream);
            size_t stream_end = min(stream_start + pages_per_stream, batch_end);
            size_t stream_pages = stream_end - stream_start;

            if (stream_pages == 0)
                continue;

            // Calculate compressed size for this stream's portion
            size_t stream_compressed_size = 0;
            for (size_t i = stream_start; i < stream_end; i++)
            {
                stream_compressed_size += compressed_data->compressed_pages[i].size;
            }

            // Allocate stream resources
            char *stream_compressed_data;
            void **stream_compressed_ptrs;
            size_t *stream_compressed_sizes;
            void **stream_uncompressed_ptrs;
            size_t *stream_uncompressed_sizes;
            nvcompStatus_t *stream_statuses;

            CHECK_CUDA(cudaMalloc(&stream_compressed_data, stream_compressed_size));
            CHECK_CUDA(cudaMalloc(&stream_compressed_ptrs, stream_pages * sizeof(void *)));
            CHECK_CUDA(cudaMalloc(&stream_compressed_sizes, stream_pages * sizeof(size_t)));
            CHECK_CUDA(cudaMalloc(&stream_uncompressed_sizes, stream_pages * sizeof(size_t)));
            CHECK_CUDA(cudaMalloc(&stream_uncompressed_ptrs, stream_pages * sizeof(void *)));
            CHECK_CUDA(cudaMalloc(&stream_statuses, stream_pages * sizeof(nvcompStatus_t)));

            // Setup and copy data
            void **host_compressed_ptrs = (void **)malloc(stream_pages * sizeof(void *));
            void **host_uncompressed_ptrs = (void **)malloc(stream_pages * sizeof(void *));
            size_t *host_compressed_sizes = (size_t *)malloc(stream_pages * sizeof(size_t));

            char *current_pos = stream_compressed_data;
            for (size_t i = 0; i < stream_pages; i++)
            {
                size_t page_idx = stream_start + i;
                host_compressed_ptrs[i] = current_pos;
                host_compressed_sizes[i] = compressed_data->compressed_pages[page_idx].size;
                host_uncompressed_ptrs[i] = *output_data + (page_idx * PAGE_SIZE);

                CHECK_CUDA(cudaMemcpyAsync(current_pos,
                                           compressed_data->compressed_pages[page_idx].data,
                                           host_compressed_sizes[i],
                                           cudaMemcpyHostToDevice,
                                           streams[stream_idx]));
                current_pos += host_compressed_sizes[i];
            }

            // Copy metadata to device
            CHECK_CUDA(cudaMemcpyAsync(stream_compressed_ptrs,
                                       host_compressed_ptrs,
                                       stream_pages * sizeof(void *),
                                       cudaMemcpyHostToDevice,
                                       streams[stream_idx]));

            CHECK_CUDA(cudaMemcpyAsync(stream_compressed_sizes,
                                       host_compressed_sizes,
                                       stream_pages * sizeof(size_t),
                                       cudaMemcpyHostToDevice,
                                       streams[stream_idx]));

            CHECK_CUDA(cudaMemcpyAsync(stream_uncompressed_ptrs,
                                       host_uncompressed_ptrs,
                                       stream_pages * sizeof(void *),
                                       cudaMemcpyHostToDevice,
                                       streams[stream_idx]));

            // Get decompressed sizes
            nvcompStatus_t status = nvcompBatchedLZ4GetDecompressSizeAsync(
                (const void **)stream_compressed_ptrs,
                stream_compressed_sizes,
                stream_uncompressed_sizes,
                stream_pages,
                streams[stream_idx]);

            if (status != nvcompSuccess)
                return NVCOMP_ERROR;

            // Allocate and perform decompression
            size_t temp_bytes;
            nvcompBatchedLZ4DecompressGetTempSize(stream_pages, PAGE_SIZE, &temp_bytes);
            void *stream_temp;
            CHECK_CUDA(cudaMalloc(&stream_temp, temp_bytes));

            status = nvcompBatchedLZ4DecompressAsync(
                (const void **)stream_compressed_ptrs,
                stream_compressed_sizes,
                stream_uncompressed_sizes,
                stream_uncompressed_sizes,
                stream_pages,
                stream_temp,
                temp_bytes,
                stream_uncompressed_ptrs,
                stream_statuses,
                streams[stream_idx]);

            if (status != nvcompSuccess)
                return NVCOMP_ERROR;

            // Record completion event
            cudaEventRecord(events[stream_idx], streams[stream_idx]);

            // Clean up stream resources
            free(host_compressed_ptrs);
            free(host_uncompressed_ptrs);
            free(host_compressed_sizes);
            cudaFree(stream_compressed_data);
            cudaFree(stream_compressed_ptrs);
            cudaFree(stream_compressed_sizes);
            cudaFree(stream_uncompressed_sizes);
            cudaFree(stream_uncompressed_ptrs);
            cudaFree(stream_temp);
            cudaFree(stream_statuses);
        }

        // Wait for batch completion
        for (int i = 0; i < NUM_STREAMS; i++)
        {
            cudaEventSynchronize(events[i]);
        }
    }

    // Final cleanup
    for (int i = 0; i < NUM_STREAMS; i++)
    {
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    return SUCCESS;
}

ErrorCode decompress_improved_naive(const CompressedData *compressed_data, char **output_data, size_t *output_size)
{
//    printf("\n=== Improved Naive Decompression ===\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    // printf("\n=== Starting Decom Version ===\n");
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
    if (!compressed_sizes)
        return MEMORY_ERROR;

    for (size_t i = 0; i < num_pages; i++)
    {
        compressed_sizes[i] = compressed_data->compressed_pages[i].size;
        total_compressed_size += compressed_sizes[i];
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Initial setup: %.3f ms\n", milliseconds);
    cudaEventRecord(start);

    // Allocate unified memory for output early
    CHECK_CUDA(cudaMallocManaged(output_data, compressed_data->original_size));

    // Add memory advice after allocation (device will access this memory)
    CHECK_CUDA(cudaMemAdvise(*output_data,
                             compressed_data->original_size,
                             cudaMemAdviseSetPreferredLocation,
                             current_device));

    // Allocate device buffers
    char *device_compressed_data;
    CHECK_CUDA(cudaMalloc(&device_compressed_data, total_compressed_size));
    void **device_compressed_ptrs;
    CHECK_CUDA(cudaMalloc(&device_compressed_ptrs, num_pages * sizeof(void *)));

    void **host_compressed_ptrs = (void **)malloc(num_pages * sizeof(void *));
    if (!host_compressed_ptrs)
    {
        cudaFree(device_compressed_data);
        free(compressed_sizes);
        return MEMORY_ERROR;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Initial allocations: %.3f ms\n", milliseconds);
    cudaEventRecord(start);

    // Copy compressed data to GPU
    char *current_pos = device_compressed_data;
    for (size_t i = 0; i < num_pages; i++)
    {
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
                               num_pages * sizeof(void *),
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
    // printf("Data transfer to GPU: %.3f ms\n", milliseconds);
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

    if (status != nvcompSuccess)
    {
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
    // printf("Get decompressed sizes: %.3f ms\n", milliseconds);
    cudaEventRecord(start);

    // Setup output pointers using unified memory
    void **device_uncompressed_ptrs;
    CHECK_CUDA(cudaMalloc(&device_uncompressed_ptrs, num_pages * sizeof(void *)));

    void **host_uncompressed_ptrs = (void **)malloc(num_pages * sizeof(void *));
    if (!host_uncompressed_ptrs)
    {
        cudaFree(device_compressed_data);
        cudaFree(device_compressed_ptrs);
        cudaFree(device_compressed_sizes);
        cudaFree(device_uncompressed_sizes);
        cudaFree(device_uncompressed_ptrs);
        free(compressed_sizes);
        free(host_compressed_ptrs);
        return MEMORY_ERROR;
    }

    for (size_t i = 0; i < num_pages; i++)
    {
        host_uncompressed_ptrs[i] = *output_data + (i * PAGE_SIZE);
    }

    CHECK_CUDA(cudaMemcpyAsync(device_uncompressed_ptrs,
                               host_uncompressed_ptrs,
                               num_pages * sizeof(void *),
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
    // printf("Output setup: %.3f ms\n", milliseconds);
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

    if (status != nvcompSuccess)
    {
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
    // printf("Decompression: %.3f ms\n", milliseconds);
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
    // printf("Memory prefetch to host: %.3f ms\n", milliseconds);
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
    // printf("Cleanup: %.3f ms\n", milliseconds);
    // printf("=== Profiling Complete ===\n\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return SUCCESS;
}

ErrorCode decompress_naive(const CompressedData *compressed_data, char **output_data, size_t *output_size)
{
//    printf("\n=== Naive Decompression ===\n");
    // Timing variables
    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // printf("\n=== Starting Detailed Profiling ===\n");
    cudaEventRecord(start);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    size_t num_pages = compressed_data->num_pages;
    *output_size = compressed_data->original_size;

    // Allocation timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Initial setup: %.3f ms\n", milliseconds);
    cudaEventRecord(start);

    // Host and device allocations
    void **host_compressed_data_per_page;
    CHECK_CUDA(cudaMallocHost(&host_compressed_data_per_page, sizeof(void *) * num_pages));
    size_t *device_compressed_numbytes_per_page;
    CHECK_CUDA(cudaMalloc(&device_compressed_numbytes_per_page, sizeof(size_t) * num_pages));
    size_t *compressed_sizes = (size_t *)malloc(num_pages * sizeof(size_t));

    if (!compressed_sizes)
    {
        cudaFreeHost(host_compressed_data_per_page);
        cudaFree(device_compressed_numbytes_per_page);
        cudaStreamDestroy(stream);
        return MEMORY_ERROR;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Initial allocations: %.3f ms\n", milliseconds);
    cudaEventRecord(start);

    // Copy compressed data to GPU
    for (size_t i = 0; i < num_pages; i++)
    {
        compressed_sizes[i] = compressed_data->compressed_pages[i].size;
        CHECK_CUDA(cudaMalloc(&host_compressed_data_per_page[i], compressed_sizes[i]));
        CHECK_CUDA(cudaMemcpy(host_compressed_data_per_page[i],
                              compressed_data->compressed_pages[i].data,
                              compressed_sizes[i],
                              cudaMemcpyHostToDevice));
    }

    CHECK_CUDA(cudaMemcpy(device_compressed_numbytes_per_page, compressed_sizes,
                          sizeof(size_t) * num_pages, cudaMemcpyHostToDevice));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Data transfer to GPU: %.3f ms\n", milliseconds);
    cudaEventRecord(start);

    free(compressed_sizes);

    // Get decompressed sizes
    size_t *device_uncompressed_numbytes_per_page;
    CHECK_CUDA(cudaMalloc(&device_uncompressed_numbytes_per_page, sizeof(size_t) * num_pages));

    nvcompStatus_t status = nvcompBatchedLZ4GetDecompressSizeAsync(
        (const void **)host_compressed_data_per_page,
        device_compressed_numbytes_per_page,
        device_uncompressed_numbytes_per_page,
        num_pages,
        stream);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Get decompressed sizes: %.3f ms\n", milliseconds);
    cudaEventRecord(start);

    if (status != nvcompSuccess)
    {
        for (size_t i = 0; i < num_pages; i++)
        {
            cudaFree(host_compressed_data_per_page[i]);
        }
        cudaFreeHost(host_compressed_data_per_page);
        cudaFree(device_compressed_numbytes_per_page);
        cudaFree(device_uncompressed_numbytes_per_page);
        cudaStreamDestroy(stream);
        return NVCOMP_ERROR;
    }

    // Temp buffer allocation
    size_t decomp_temp_bytes;
    nvcompBatchedLZ4DecompressGetTempSize(num_pages, PAGE_SIZE, &decomp_temp_bytes);
    void *device_temp_ptr;
    CHECK_CUDA(cudaMalloc(&device_temp_ptr, decomp_temp_bytes));

    nvcompStatus_t *device_statuses;
    CHECK_CUDA(cudaMalloc(&device_statuses, sizeof(nvcompStatus_t) * num_pages));

    size_t *device_actual_uncompressed_numbytes_per_page;
    CHECK_CUDA(cudaMalloc(&device_actual_uncompressed_numbytes_per_page, sizeof(size_t) * num_pages));

    char *device_output_data;
    CHECK_CUDA(cudaMalloc(&device_output_data, compressed_data->original_size));

    void **device_uncompressed_data_per_page;
    void **host_uncompressed_data_per_page;
    CHECK_CUDA(cudaMalloc(&device_uncompressed_data_per_page, sizeof(void *) * num_pages));
    CHECK_CUDA(cudaMallocHost(&host_uncompressed_data_per_page, sizeof(void *) * num_pages));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Temp buffer and output allocations: %.3f ms\n", milliseconds);
    cudaEventRecord(start);

    // Setup output pointers
    for (size_t i = 0; i < num_pages; i++)
    {
        host_uncompressed_data_per_page[i] = device_output_data + PAGE_SIZE * i;
    }
    CHECK_CUDA(cudaMemcpy(device_uncompressed_data_per_page, host_uncompressed_data_per_page,
                          sizeof(void *) * num_pages, cudaMemcpyHostToDevice));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Output pointer setup: %.3f ms\n", milliseconds);
    cudaEventRecord(start);

    // Actual decompression
    status = nvcompBatchedLZ4DecompressAsync(
        (const void **)host_compressed_data_per_page,
        device_compressed_numbytes_per_page,
        device_uncompressed_numbytes_per_page,
        device_actual_uncompressed_numbytes_per_page,
        num_pages,
        device_temp_ptr,
        decomp_temp_bytes,
        device_uncompressed_data_per_page,
        device_statuses,
        stream);

    cudaStreamSynchronize(stream); // Make sure decompression is complete before timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Actual decompression: %.3f ms\n", milliseconds);
    cudaEventRecord(start);

    if (status != nvcompSuccess)
    {
        for (size_t i = 0; i < num_pages; i++)
        {
            cudaFree(host_compressed_data_per_page[i]);
        }
        cudaFreeHost(host_compressed_data_per_page);
        cudaFree(device_compressed_numbytes_per_page);
        cudaFree(device_uncompressed_numbytes_per_page);
        cudaFree(device_temp_ptr);
        cudaFree(device_statuses);
        cudaFree(device_actual_uncompressed_numbytes_per_page);
        cudaFree(device_uncompressed_data_per_page);
        cudaFree(device_output_data);
        cudaFreeHost(host_uncompressed_data_per_page);
        cudaStreamDestroy(stream);
        return NVCOMP_ERROR;
    }

    // Allocate and copy result to host
    *output_data = (char *)malloc(compressed_data->original_size);
    if (!*output_data)
    {
        for (size_t i = 0; i < num_pages; i++)
        {
            cudaFree(host_compressed_data_per_page[i]);
        }
        cudaFreeHost(host_compressed_data_per_page);
        cudaFree(device_compressed_numbytes_per_page);
        cudaFree(device_uncompressed_numbytes_per_page);
        cudaFree(device_temp_ptr);
        cudaFree(device_statuses);
        cudaFree(device_actual_uncompressed_numbytes_per_page);
        cudaFree(device_uncompressed_data_per_page);
        cudaFree(device_output_data);
        cudaFreeHost(host_uncompressed_data_per_page);
        cudaStreamDestroy(stream);
        return MEMORY_ERROR;
    }

    CHECK_CUDA(cudaMemcpy(*output_data, device_output_data,
                          compressed_data->original_size, cudaMemcpyDeviceToHost));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Copy result back to host: %.3f ms\n", milliseconds);
    cudaEventRecord(start);

    // Cleanup
    for (size_t i = 0; i < num_pages; i++)
    {
        cudaFree(host_compressed_data_per_page[i]);
    }
    cudaFreeHost(host_compressed_data_per_page);
    cudaFree(device_compressed_numbytes_per_page);
    cudaFree(device_uncompressed_numbytes_per_page);
    cudaFree(device_temp_ptr);
    cudaFree(device_statuses);
    cudaFree(device_actual_uncompressed_numbytes_per_page);
    cudaFree(device_uncompressed_data_per_page);
    cudaFree(device_output_data);
    cudaFreeHost(host_uncompressed_data_per_page);

    cudaStreamDestroy(stream);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Cleanup: %.3f ms\n", milliseconds);
    // printf("=== Profiling Complete ===\n\n");

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
