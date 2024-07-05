#ifndef PQ_DIST_H
#define PQ_DIST_H

#include <faiss/IndexPQ.h>
#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include <iostream>
#include <vector>
#include <cstdlib>  // For rand() and srand()
#include <ctime>    // For time()
#include <bits/stdc++.h>
#include "../hnswlib/hnswlib.h"
#include "dir_vector.h"
using namespace std;
class PQDist {
public:
    PQDist() = default;
    ~PQDist() = default;

    PQDist(int _d, int _m, int _nbits);
    int d, m, nbits;
    int code_nums;
    int d_pq;
    std::vector<uint8_t> codes;
    std::vector<float> centroids;
    std::unique_ptr<faiss::IndexPQ> indexPQ;
    std::unique_ptr<hnswlib::SpaceInterface<float> > space;

    void train(int N, std::vector<float> &xb);

    std::vector<int> get_centroids_id(int id);
    float* get_centroid_data(int quantizer, int code_id);

    float calc_dist(int d, float *vec1, float *vec2);
    float calc_dist_pq(int data_id, float *qdata, bool use_cache);


    std::vector<float> pq_dist_cache;
    void clear_pq_dist_cache();

    std::vector<float> qdata;
    bool use_cache;
    void load_query_data(const float *_qdata, bool _use_cache);
    void load_query_data_and_cache(const float *_qdata) ;
    float calc_dist_pq_(int data_id, float *qdata, bool use_cache);
    float calc_dist_pq_loaded(int data_id);
    float calc_dist_pq_loaded_(int data_id);
    void load(std::string filename);
    vector<int> encode_query(float* query);
    vector<vector<vector<float>>> distance_table;
    void construct_distance_table();
    float calc_dist_pq_from_table(int data_id, vector<int>& qids);
    float calc_dist_pq_simd(int data_id, float *qdata, bool use_cache);
    float calc_dist_pq_loaded_simd(int data_id);
    
};

#endif // !PQ_DIST_H