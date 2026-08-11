#include <iostream>
#include "reduce_graph.hpp"
#include "timer.hpp"

int main(int argc, char **argv) {
    std::vector<size_t> values(100'000'000);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = i + 1;
    }
    timer_start(reduce_shared_task);
    ReduceGraph<size_t> graph;
    graph.executeGraph();
    graph.pushData(std::make_shared<std::vector<size_t>>(std::move(values)));
    graph.finishPushingData();
    auto result = graph.getBlockingResult();
    std::cout << *std::get<std::shared_ptr<size_t>>(*result) << std::endl;
    graph.waitForTermination();
    timer_end(reduce_shared_task);
    timer_report(reduce_shared_task);
    return 0;
}
