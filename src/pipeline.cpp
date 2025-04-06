// pipeline for bwa-mem operation.

#include "pipeline.h"
#include "concurrentqueue.h"
#include <queue>
#include <vector>
#include <thread>


void pipeline(pipeline_aux_t *aux)
{
    moodycamel::ConcurrentQueue<parsed_chunk> dispatch_queue;
    std::priority_queue<
        aligned_chunk, std::vector<aligned_chunk>, writequeue_compare
        > write_queue;

    std::vector<std::thread> load_threads, dispatch_threads;

    for(int t=0; t<aux->load_thread_cnt; t++) {
        load_threads.emplace_back(worker_load_and_parse, aux->fd_input,
               aux->load_chunk_bytes, t, aux->load_thread_cnt, std::ref(dispatch_queue));
    }
    for(int t=0; t<aux->dispatch_thread_cnt; t++) {
        dispatch_threads.emplace_back(worker_dispatch, std::ref(dispatch_queue),
                aux, std::ref(write_queue));
    }
    std::thread write_thread(worker_write, std::ref(write_queue), aux->samout);


    for (auto &th : load_threads)
        th.join();
    for (auto &th : dispatch_threads)
        th.join();
    write_thread.join();
}


void worker_load_and_parse(
        int fd_input,
        long load_chunk_bytes,
        int tid,
        int thread_cnt,
        moodycamel::ConcurrentQueue<parsed_chunk> &dispatch_queue
        )
{
    //dispatch_queue.enqueue(25);

    //bool found = dispatch_queue.try_dequeue(item);
}

void worker_dispatch(
        moodycamel::ConcurrentQueue<parsed_chunk> &dispatch_queue,
        pipeline_aux_t *aux,
        std::priority_queue<
            aligned_chunk, std::vector<aligned_chunk>, writequeue_compare
        > &write_queue
        )
{

}

void worker_write(
        std::priority_queue<
            aligned_chunk, std::vector<aligned_chunk>, writequeue_compare
        > &write_queue,
        std::ostream *samout
        )
{

}
