#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <thread>
#include <vector>

// Thread function: processes file data from [start, end)
void process_chunk(const char* data, size_t start, size_t end) {
    // Example processing: count newline characters
    size_t count = 0;
    for (size_t i = start; i < end; ++i) {
        if (data[i] == '\n') {
            ++count;
        }
    }
    std::cout << "Processed chunk [" << start << ", " << end << "): " 
        << count << " newline(s) found.\n";
}

int main() {
    const char* filename = "example.txt";
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        perror("open");
        return 1;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("fstat");
        close(fd);
        return 1;
    }
    size_t filesize = sb.st_size;

    // Map the entire file into memory.
    char* data = static_cast<char*>(mmap(nullptr, filesize, PROT_READ, MAP_PRIVATE, fd, 0));
    if (data == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }
    close(fd);  // File descriptor no longer needed after mapping.

    // Set up multithreading.
    const int num_threads = 4;
    std::vector<std::thread> threads;
    size_t chunk_size = filesize / num_threads;

    for (int i = 0; i < num_threads; ++i) {
        size_t start = i * chunk_size;
        // Ensure the last thread processes any remaining bytes.
        size_t end = (i == num_threads - 1) ? filesize : start + chunk_size;
        threads.emplace_back(process_chunk, data, start, end);
    }

    // Wait for all threads to finish.
    for (auto& t : threads) {
        t.join();
    }

    // Unmap the file.
    if (munmap(data, filesize) == -1) {
        perror("munmap");
        return 1;
    }

    return 0;
}

}
}
}
}
}
}
}
}
}
}
