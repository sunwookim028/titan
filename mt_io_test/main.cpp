// 
// Multiple producers & in-order single consumer.
//
#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <map>
#include <vector>

// Record structure
struct Record {
    int id;
    std::string data;
};

// Shared variables
std::mutex mtx;
std::condition_variable cv;
std::map<int, Record> recordBuffer; // Buffer keyed by record id
int nextExpectedId = 0;
bool producersDone = false;

// Producer function
void producerFunction(int startId, int count) {
    for (int i = 0; i < count; ++i) {
        Record r;
        r.id = startId + i;
        r.data = "Record data " + std::to_string(r.id);
        {
            std::lock_guard<std::mutex> lock(mtx);
            recordBuffer[r.id] = r;
        }
        cv.notify_one();
        // Simulate work (optional)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

// Consumer function
void consumerFunction(const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output file.\n";
        return;
    }

    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        // Wait until there is some data or producers have finished
        cv.wait(lock, [] { return !recordBuffer.empty() || producersDone; });
        
        // Check if the next expected record is available
        auto it = recordBuffer.find(nextExpectedId);
        if (it != recordBuffer.end()) {
            // Write the record to file
            outfile << "ID: " << it->second.id 
                    << " Data: " << it->second.data << "\n";
            recordBuffer.erase(it);
            ++nextExpectedId;
        }
        // If producers are done and the buffer doesn't contain the next record, break.
        else if (producersDone) {
            break;
        }
    }
    outfile.close();
}

int main() {
    // Example: two producers generating 10 records each.
    std::vector<std::thread> producers;
    producers.emplace_back(producerFunction, 0, 10);
    producers.emplace_back(producerFunction, 10, 10);

    std::thread consumer(consumerFunction, "output.txt");

    // Wait for all producers to finish
    for (auto& p : producers) {
        p.join();
    }
    {
        std::lock_guard<std::mutex> lock(mtx);
        producersDone = true;
    }
    cv.notify_all();

    consumer.join();
    return 0;
}
