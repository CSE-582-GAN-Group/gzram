
#include "naive.cuh"
#include <time.h>
#include <lz4.h>
#include <lz4hc.h>

typedef struct {
    char* data;
    size_t size;
} CPUCompressedData;

ErrorCode cpu_compress(const char* input_data, size_t input_size, CPUCompressedData* output) {
    // Calculate max compressed size needed
    int max_dst_size = LZ4_compressBound(input_size);
    
    // Initialize output structure
    output->data = (char*)malloc(max_dst_size);
    
    if (!output->data) {
        return MEMORY_ERROR;
    }

    output->size = LZ4_compress_default(input_data, 
                                      output->data, 
                                      input_size, 
                                      max_dst_size);
    
    if (output->size <= 0) {
        free(output->data);
        return COMPRESSION_ERROR;
    }
    
    return SUCCESS;
}

ErrorCode cpu_decompress(const CPUCompressedData* input, char** output_data, size_t original_size) {
    *output_data = (char*)malloc(original_size);
    if (!*output_data) {
        return MEMORY_ERROR;
    }

    int decompressed_size = LZ4_decompress_safe(input->data, *output_data, input->size, original_size);
    
    if (decompressed_size != original_size) {
        free(*output_data);
        *output_data = NULL;
        return DECOMPRESSION_ERROR;
    }
    
    return SUCCESS;
}

void free_cpu_compressed_data(CPUCompressedData* data) {
    if (data) {
        free(data->data);
        data->data = NULL;
        data->size = 0;
    }
}

int main(int argc, char *argv[])
{
    // Check if filename was provided
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    // Open the file
    FILE *file = fopen(argv[1], "rb");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", argv[1]);
        return 1;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    size_t data_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Allocate memory for the data
    char *original_data = (char *)malloc(data_size);
    if (!original_data) {
        fprintf(stderr, "Failed to allocate memory for original data\n");
        fclose(file);
        return 1;
    }

    // Read the file
    size_t bytes_read = fread(original_data, 1, data_size, file);
    fclose(file);

    if (bytes_read != data_size) {
        fprintf(stderr, "Failed to read entire file\n");
        free(original_data);
        return 1;
    }

    print_data(original_data, data_size, "Original Data");

    // Variables for timing
    struct timespec start, end;
    double cpu_compress_time, cpu_decompress_time, gpu_compress_time, gpu_decompress_time;

    // CPU Compression
    CPUCompressedData cpu_compressed = {0};
    clock_gettime(CLOCK_MONOTONIC, &start);
    ErrorCode error = cpu_compress(original_data, data_size, &cpu_compressed);
    clock_gettime(CLOCK_MONOTONIC, &end);
    cpu_compress_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    if (error != SUCCESS) {
        fprintf(stderr, "CPU Compression failed with error code: %d\n", error);
        free(original_data);
        return 1;
    }

    // GPU Compression
    CompressedData *gpu_compressed = NULL;
    clock_gettime(CLOCK_MONOTONIC, &start);
    error = compress(original_data, data_size, &gpu_compressed);
    clock_gettime(CLOCK_MONOTONIC, &end);
    gpu_compress_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    if (error != SUCCESS) {
        fprintf(stderr, "GPU Compression failed with error code: %d\n", error);
        free(original_data);
        free_cpu_compressed_data(&cpu_compressed);
        return 1;
    }

    // Print compression results
    printf("\nCompression Results:\n");
    printf("Original size: %zu bytes\n", data_size);
    printf("CPU Compressed size: %zu bytes (ratio: %.2f)\n", 
           cpu_compressed.size, (float)data_size / cpu_compressed.size);
    
    size_t gpu_total_size = 0;
    for (size_t i = 0; i < gpu_compressed->num_pages; i++) {
        gpu_total_size += gpu_compressed->compressed_pages[i].size;
    }
    printf("GPU Compressed size: %zu bytes (ratio: %.2f)\n", 
           gpu_total_size, (float)data_size / gpu_total_size);

    // CPU Decompression
    char *cpu_decompressed = NULL;
    clock_gettime(CLOCK_MONOTONIC, &start);
    error = cpu_decompress(&cpu_compressed, &cpu_decompressed, data_size);
    clock_gettime(CLOCK_MONOTONIC, &end);
    cpu_decompress_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    if (error != SUCCESS) {
        fprintf(stderr, "CPU Decompression failed with error code: %d\n", error);
        free(original_data);
        free_cpu_compressed_data(&cpu_compressed);
        free_compressed_data(gpu_compressed);
        return 1;
    }

    // GPU Decompression
    char *gpu_decompressed = NULL;
    size_t decompressed_size = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    error = decompress(gpu_compressed, &gpu_decompressed, &decompressed_size);
    clock_gettime(CLOCK_MONOTONIC, &end);
    gpu_decompress_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    // Print timing results
    printf("\nTiming Results:\n");
    printf("CPU Compression time: %f seconds\n", cpu_compress_time);
    printf("GPU Compression time: %f seconds\n", gpu_compress_time);
    printf("CPU Decompression time: %f seconds\n", cpu_decompress_time);
    printf("GPU Decompression time: %f seconds\n", gpu_decompress_time);

    // Verify both decompressed results
    printf("\nVerification Results:\n");
    printf("CPU Decompression: %s\n", 
           (memcmp(original_data, cpu_decompressed, data_size) == 0) ? "SUCCESS" : "FAILURE");
    printf("GPU Decompression: %s\n", 
           (memcmp(original_data, gpu_decompressed, data_size) == 0) ? "SUCCESS" : "FAILURE");

    // Cleanup
    free(original_data);
    free(cpu_decompressed);
    free(gpu_decompressed);
    free_cpu_compressed_data(&cpu_compressed);
    free_compressed_data(gpu_compressed);

    return 0;
}