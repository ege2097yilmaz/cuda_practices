#include "prefix_sum.h"
#include <iostream>
#include <vector>

int main(){
    std::vector<int>  input = {1, 2, 3, 4, 5, 6, 7, 8};

    for(const auto& val : input) std::cout << val << " ";
    std::cout << std::endl;

    std::vector<int> output = paralel_prefix_sum(input);

    std::cout << "output";

    for(const auto& val : output) std::cout << val << " ";
    std::cout << std::endl;

    return 0;
}