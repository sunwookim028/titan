#include <fcntl.h>      // for open()
#include <unistd.h>     // for read(), close()
#include <sys/types.h>
#include <sys/stat.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>


#define TIMER_INIT();\
        std::chrono::high_resolution_clock::time_point start, end;\
        std::chrono::duration<long long, std::micro> duration;

#define TIMER_START();\
    start = std::chrono::high_resolution_clock::now();

#define TIMER_END(event_name);\
    end = std::chrono::high_resolution_clock::now();\
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);\
    std::cerr << "* " << event_name << ": " << duration.count() / 1000 << " ms" << std::endl;


struct RecordBatch {
    uint64_t file_offset;              // File offset of the first record in this batch.
    int batch_size = 0;                // Number of records in this batch.
    std::vector<int> seq_offsets;      // Offsets into seq_concat for each record.
    std::vector<int> name_offsets;     // Offsets into name_concat for each record.
    std::vector<int> qual_offsets;     // Offsets into qual_concat for each record.
    std::string seq_concat;            // Concatenated SEQ strings.
    std::string name_concat;           // Concatenated NAME strings.
    std::string qual_concat;           // Concatenated QUAL strings.
};

// loadBatch reads the next batch (num_records) from file descriptor fd,
// using low-level read() and splits the batch into lines, then parses
// records concurrently using num_threads threads.
RecordBatch loadBatch(int fd, int num_records, int num_threads, char *buffer, size_t bufsize) {

    TIMER_INIT();
    TIMER_START();

    RecordBatch finalBatch;
    // Record the file offset at the start of the batch.
    finalBatch.file_offset = static_cast<uint64_t>(lseek(fd, 0, SEEK_CUR));

    // We expect each record to span 4 lines.
    const int total_lines_needed = num_records * 4;
    std::vector<std::string> lines;
    lines.reserve(total_lines_needed);

    std::string residual;  // To hold a partial line from the previous read.
    int lines_read = 0;
    bool finished = false;

    // Read until we have at least the required number of lines or reach EOF.
    while (lines_read < total_lines_needed) {
        ssize_t bytes_read = read(fd, buffer, bufsize);
        if (bytes_read < 0) {
            std::cerr << "Error reading file: " << std::strerror(errno) << "\n";
            break;
        }
        if (bytes_read == 0) {
            // End of file reached.
            break;
        }
        std::string chunk(buffer, bytes_read);
        if (!residual.empty()) {
            chunk = residual + chunk;
            residual.clear();
        }
        size_t start = 0;
        size_t pos = 0;
        // Split the chunk by newline.
        while ((pos = chunk.find('\n', start)) != std::string::npos) {
            std::string line = chunk.substr(start, pos - start);
            lines.push_back(line);
            ++lines_read;
            start = pos + 1;
            if (lines_read >= total_lines_needed){
                finished = true;
                break;
            }
        }
        if (finished) {
            // If there are extra bytes in the chunk beyond what we needed,
            // push them back onto the file descriptor.
            int extra = chunk.size() - start;
            if (extra > 0) {
                // Since read() advanced the file offset by bytes_read,
                // we lseek back by the number of unprocessed bytes.
                if (lseek(fd, -extra, SEEK_CUR) == (off_t)-1) {
                    std::cerr << "lseek error: " << std::strerror(errno) << "\n";
                }
            }
            break;
        }
        // Save any partial line.
        if (start < chunk.size() && lines_read < total_lines_needed) {
            residual = chunk.substr(start);
        }
    }
    // If there's a leftover (without a newline), add it.
    if (!residual.empty() && lines_read < total_lines_needed) {
        lines.push_back(residual);
        ++lines_read;
        residual.clear();
    }

    // Determine how many complete records we have.
    int total_records = lines.size() / 4;
    if (total_records == 0)
        return finalBatch;  // No records loaded.

    TIMER_END("loading");

    TIMER_START();

    // Prepare for multi-threaded parsing.
    int records_per_thread = total_records / num_threads;
    int remainder = total_records % num_threads;
    std::vector<RecordBatch> localBatches(num_threads);
    std::vector<std::thread> threads;

    // Each thread will parse a contiguous segment of records.
    auto parseSegment = [&](int threadIndex, int startRecord, int endRecord) {
        RecordBatch localBatch;
        for (int i = startRecord; i < endRecord; i++) {
            int base = i * 4;
            // Line 0: header in format "NAME AUX length=LEN"
            std::istringstream iss(lines[base].substr(1)); // skipping '@'.
            std::string name;
            if (!(iss >> name)) {
                std::cerr << "Parsing error in header at record " << i << "\n";
                continue;
            }
            // Save NAME, SEQ (line1) and QUAL (line3).
            localBatch.name_offsets.push_back(localBatch.name_concat.size());
            localBatch.seq_offsets.push_back(localBatch.seq_concat.size());
            localBatch.qual_offsets.push_back(localBatch.qual_concat.size());
            localBatch.name_concat.append(name);
            localBatch.seq_concat.append(lines[base + 1]);  // SEQ field.
            localBatch.qual_concat.append(lines[base + 3]);   // QUAL field.
            localBatch.batch_size++;
        }
        localBatches[threadIndex] = std::move(localBatch);
    };

    int currentRecord = 0;
    for (int t = 0; t < num_threads; t++) {
        int count = records_per_thread + (t < remainder ? 1 : 0);
        int startRecord = currentRecord;
        int endRecord = startRecord + count;
        threads.emplace_back(parseSegment, t, startRecord, endRecord);
        currentRecord = endRecord;
    }
    for (auto &th : threads)
        th.join();

    TIMER_END("parsing");

    TIMER_START();

    // Merge all local batches into finalBatch (maintaining record order).
    for (const auto& local : localBatches) {
        int name_offset_adjust = finalBatch.name_concat.size();
        int seq_offset_adjust = finalBatch.seq_concat.size();
        int qual_offset_adjust = finalBatch.qual_concat.size();

        for (int off : local.name_offsets)
            finalBatch.name_offsets.push_back(off + name_offset_adjust);
        finalBatch.name_concat.append(local.name_concat);

        for (int off : local.seq_offsets)
            finalBatch.seq_offsets.push_back(off + seq_offset_adjust);
        finalBatch.seq_concat.append(local.seq_concat);

        for (int off : local.qual_offsets)
            finalBatch.qual_offsets.push_back(off + qual_offset_adjust);
        finalBatch.qual_concat.append(local.qual_concat);

        finalBatch.batch_size += local.batch_size;
    }

    TIMER_END("merging");

    return finalBatch;
}

int main(int argc, char* argv[]) {
    TIMER_INIT();
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <filename> <num_threads> <batch size> <loading bufsize>\n";
        return 1;
    }
    int num_threads = std::stoi(argv[2]);
    // Set the desired number of records per batch (e.g., 100,000 records).
    int batch_records = std::stoi(argv[3]);

    TIMER_START();

    // Open file with low-level I/O.
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) {
        std::cerr << "Error opening file: " << std::strerror(errno) << "\n";
        return 1;
    }

    TIMER_END("file open");

    std::string prefix = argv[1];
    std::string filename = prefix + ".out";
    std::ofstream outFile(filename);
    if(!outFile){
        std::cerr << "Error opening file: " << filename << std::endl;
    }

    size_t bufsize = std::stoi(argv[4]);
    char *buffer = new char[bufsize];
    if (buffer == 0) {
        std::cerr << "Loading buffer malloc failed." << std::endl;
        return 1;
    }

    while (true) {
        // Load one batch at a time.
        RecordBatch batch = loadBatch(fd, batch_records, num_threads, buffer, bufsize);
        if (batch.batch_size == 0)
            break;
        std::cout << "Loaded batch with " << batch.batch_size << " records.\n";

        // At this point, the batch would be handed off to computation/storage threads.
        // In a complete system you might use a queue to overlap I/O and processing.
        // outFile << batch.file_offset << std::endl;
        // outFile << batch.seq_concat << std::endl;
         outFile << batch.name_concat << std::endl;
        // outFile << batch.qual_concat << std::endl;
    }

    outFile.close();

    close(fd);
    return 0;
}
