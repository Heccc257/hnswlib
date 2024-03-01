#pragma once

#include <bits/stdc++.h>

namespace DATALOADER {
class DataLoader {

public:
    DataLoader(std::string data_type_name, uint32_t _max_elements, std::string _data_path);
    ~DataLoader();

    const void *point_data(int id) {
        if (id >= elements) {
            std::cerr << "only have " << elements << " points\n";
            return nullptr;
        }
        return reinterpret_cast<const void*>(data + id * dim * data_type_len);
    }
    void print_point_data_int8(int id) {
        const uint8_t *point = reinterpret_cast<const uint8_t*>(point_data(id));
        for (int i = 0; i < dim; i++)
            std::cout << (uint32_t)*(point + i) << ' '; 
            std::cout << '\n';
    }
    uint32_t get_elements();
    uint32_t get_dim();
    void free_data();
private:

    void *data;
    int dim;
    uint32_t elements;
    uint64_t data_type_len;
    uint64_t tot_data_size;
    std::string data_path;
};
}