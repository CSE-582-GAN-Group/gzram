#include <naive.cuh>

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

ErrorCode compress(const char *input_data, size_t in_bytes, CompressedData **output)
{
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

ErrorCode decompress(const CompressedData *compressed_data, char **output_data, size_t *output_size)
{
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    size_t num_pages = compressed_data->num_pages;
    *output_size = compressed_data->original_size;

    // Allocate and copy compressed data to device
    void **host_compressed_data_per_page;
    CHECK_CUDA(cudaMallocHost(&host_compressed_data_per_page, sizeof(void *) * num_pages));

    // Need an array of sizes for the API
    size_t *device_compressed_numbytes_per_page;
    CHECK_CUDA(cudaMalloc(&device_compressed_numbytes_per_page, sizeof(size_t) * num_pages));

    // Create temporary array of sizes and copy to device
    size_t *compressed_sizes = (size_t *)malloc(num_pages * sizeof(size_t));
    if (!compressed_sizes)
    {
        cudaFreeHost(host_compressed_data_per_page);
        cudaFree(device_compressed_numbytes_per_page);
        cudaStreamDestroy(stream);
        return MEMORY_ERROR;
    }

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

    // Get temp buffer size and allocate it
    size_t decomp_temp_bytes;
    nvcompBatchedLZ4DecompressGetTempSize(num_pages, PAGE_SIZE, &decomp_temp_bytes);
    void *device_temp_ptr;
    CHECK_CUDA(cudaMalloc(&device_temp_ptr, decomp_temp_bytes));

    nvcompStatus_t *device_statuses;
    CHECK_CUDA(cudaMalloc(&device_statuses, sizeof(nvcompStatus_t) * num_pages));

    size_t *device_actual_uncompressed_numbytes_per_page;
    CHECK_CUDA(cudaMalloc(&device_actual_uncompressed_numbytes_per_page, sizeof(size_t) * num_pages));

    // Allocate output buffer
    char *device_output_data;
    CHECK_CUDA(cudaMalloc(&device_output_data, compressed_data->original_size));

    // Setup output pointers
    void **device_uncompressed_data_per_page;
    void **host_uncompressed_data_per_page;
    CHECK_CUDA(cudaMalloc(&device_uncompressed_data_per_page, sizeof(void *) * num_pages));
    CHECK_CUDA(cudaMallocHost(&host_uncompressed_data_per_page, sizeof(void *) * num_pages));
    for (size_t i = 0; i < num_pages; i++)
    {
        host_uncompressed_data_per_page[i] = device_output_data + PAGE_SIZE * i;
    }
    CHECK_CUDA(cudaMemcpy(device_uncompressed_data_per_page, host_uncompressed_data_per_page,
                          sizeof(void *) * num_pages, cudaMemcpyHostToDevice));

    // Decompress the data
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

    // Cleanup all allocated resources
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

