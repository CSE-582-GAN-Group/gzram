
#include "naive.cuh"

int main()
{
    // Generate sample data with patterns (more compressible)
    const size_t data_size = 1000000;
    char *original_data = (char *)malloc(data_size);
    if (!original_data)
    {
        fprintf(stderr, "Failed to allocate memory for original data\n");
        return 1;
    }

    // Create repeating pattern
    const char pattern[] = "hi, this is a test string that will be repeated many times";
    const size_t pattern_length = strlen(pattern);

    for (size_t i = 0; i < data_size; i++)
    {
        original_data[i] = pattern[i % pattern_length];
    }

    print_data(original_data, data_size, "Original Data");

    // Compress the data
    CompressedData *compressed = NULL;
    ErrorCode error = compress(original_data, data_size, &compressed);

    if (error != SUCCESS)
    {
        fprintf(stderr, "Compression failed with error code: %d\n", error);
        free(original_data);
        return 1;
    }

    // Calculate total compressed size
    size_t total_compressed_size = 0;
    for (size_t i = 0; i < compressed->num_pages; i++)
    {
        total_compressed_size += compressed->compressed_pages[i].size;
    }

    printf("\nCompression Results:\n");
    printf("Original size: %zu bytes\n", data_size);
    printf("Compressed size: %zu bytes\n", total_compressed_size);
    printf("Compression ratio: %.2f\n", (float)data_size / total_compressed_size);

    // Example of using the new helper function
    // First, create arrays of our compressed data
    const char **page_data_array = (const char **)malloc(compressed->num_pages * sizeof(char *));
    size_t *page_sizes_array = (size_t *)malloc(compressed->num_pages * sizeof(size_t));

    for (size_t i = 0; i < compressed->num_pages; i++)
    {
        page_data_array[i] = compressed->compressed_pages[i].data;
        page_sizes_array[i] = compressed->compressed_pages[i].size;
    }

    // Create a new CompressedData using our helper function
    CompressedData *compressed_copy = create_compressed_data_from_arrays(
        page_data_array,
        page_sizes_array,
        compressed->num_pages,
        compressed->original_size);

    free(page_data_array);
    free(page_sizes_array);

    // Decompress the data
    char *decompressed_data = NULL;
    size_t decompressed_size = 0;
    error = decompress(compressed, &decompressed_data, &decompressed_size);
    // error = decompress(compressed_copy, &decompressed_data, &decompressed_size);

    if (error != SUCCESS)
    {
        fprintf(stderr, "Decompression failed with error code: %d\n", error);
        free(original_data);
        free_compressed_data(compressed);
        free_compressed_data(compressed_copy);
        return 1;
    }

    print_data(decompressed_data, decompressed_size, "Decompressed Data");

    // Verify the data
    int match = (decompressed_size == data_size &&
                 memcmp(original_data, decompressed_data, data_size) == 0);

    printf("\nVerification: %s\n", match ? "SUCCESS" : "FAILURE");

    // Cleanup
    free(original_data);
    free(decompressed_data);
    free_compressed_data(compressed);
    free_compressed_data(compressed_copy);

    return 0;
}