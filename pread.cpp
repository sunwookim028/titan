#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>
#include <string>
#include <algorithm>

Structure to hold a batch of FASTQ data as lines.
struct RecordBatch {
    off_t file_offset;               // Starting file offset of this batch.
    std::vector<std::string> lines;  // The lines in this batch (to be parsed into FASTQ records).
};

// Loads a batch using parallel pread. Each thread reads its piece without boundary checking.
// After merging, the function performs boundary detection to ensure only complete FASTQ records are returned.
// The global offset is updated accordingly.
RecordBatch loadBatchWithBoundaryParallel(int fd, off_t &offset, size_t batchSize, int num_threads) {
    RecordBatch batch;
    batch.file_offset = offset;

    // Prepare to split the batchSize into nearly equal segments.
    std::vector<std::string> pieces(num_threads);
    size_t baseSegmentSize = batchSize / num_threads;
    size_t remainder = batchSize % num_threads;
    std::vector<std::thread> threads;

    // Spawn threads to read their segments concurrently using pread.
    for (int i = 0; i < num_threads; ++i) {
        // Distribute any remainder bytes among the first few threads.
        size_t segSize = baseSegmentSize + (i < remainder ? 1 : 0);
        off_t segOffset = offset;
        // Compute the offset for the current thread:
        // It is the starting offset plus the sum of sizes of previous segments.
        for (int j = 0; j < i; ++j) {
            segOffset += baseSegmentSize + (j < remainder ? 1 : 0);
        }

        threads.emplace_back([fd, segOffset, segSize, &pieces, i]() {
                std::string localBuffer;
                localBuffer.resize(segSize);
                ssize_t bytesRead = pread(fd, &localBuffer[0], segSize, segOffset);
                if (bytesRead < 0) {
                    std::cerr << "pread error: " << std::strerror(errno) << "\n";
                    localBuffer.clear();
                } else {
                    localBuffer.resize(bytesRead);
                }
                pieces[i] = std::move(localBuffer);
                });
    }

    // Wait for all threads to complete.
    for (auto &t : threads) {
        t.join();
    }

    // Merge the pieces in order.
    std::string merged;
    merged.reserve(batchSize);
    for (const auto &piece : pieces)
        merged.append(piece);

    // Perform boundary checking:
    // For FASTQ, each record starts with '@'. We scan backwards for a line starting with '@'
    // (i.e., either at the very beginning or following a newline).
    size_t boundary = std::string::npos;
    for (size_t pos = merged.size(); pos > 0; --pos) {
        if (merged[pos - 1] == '@') {
            if (pos == 1 || merged[pos - 2] == '\n') {
                boundary = pos - 1;
                break;
            }
        }
    }
    // If no boundary is found, we use the entire merged buffer.
    if (boundary == std::string::npos)
        boundary = merged.size();

    // Update the global offset: advance by the number of bytes that form complete records.
    offset += boundary;

    // Extract the valid batch data.
    std::string batchData = merged.substr(0, boundary);

    // Split the batchData into lines.
    size_t start = 0;
    size_t pos;
    while ((pos = batchData.find('\n', start)) != std::string::npos) {
        batch.lines.push_back(batchData.substr(start, pos - start));
        start = pos + 1;
    }
    if (start < batchData.size())
        batch.lines.push_back(batchData.substr(start));

    return batch;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <filename> <batch_size_in_bytes> <num_threads>\n";
        return 1;
    }
    const char* filename = argv[1];
    size_t batchSize = std::stoul(argv[2]);
    int num_threads = std::stoi(argv[3]);

    // Open the file.
    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        std::cerr << "Error opening file: " << std::strerror(errno) << "\n";
        return 1;
    }

    off_t currentOffset = 0;
    struct stat st;
    if (fstat(fd, &st) < 0) {
        std::cerr << "fstat error: " << std::strerror(errno) << "\n";
        close(fd);
        return 1;
    }
    off_t fileSize = st.st_size;

    // Read batches until end-of-file.
    while (currentOffset < fileSize) {
        RecordBatch batch = loadBatchWithBoundaryParallel(fd, currentOffset, batchSize, num_threads);
        if (batch.lines.empty())
            break;
        std::cout << "Loaded batch starting at offset " << batch.file_offset
            << " with " << batch.lines.size() << " lines." << std::endl;

        // At this point, batch.lines can be parsed into FASTQ records (each 4 lines).
        // For demonstration, print the first record header if it exists.
        if (!batch.lines.empty() && !batch.lines[0].empty() && batch.lines[0][0] == '@') {
            std::cout << "First record header: " << batch.lines[0] << std::endl;
        }
    }

    close(fd);
    return 0;
}
