#pragma once
#include <bits/stdc++.h>

#include "data_loader.h"

#include "../hnswlib/hnswlib.h"
#include "calc_group_truth.h"
#include "recall_test.h"
#include "ArgParser.h"
#include "config.h"
#include "dir_vector.h"

using namespace std;

template<typename dist_t>
class Tester {
public:
    Tester(
        CommandLineOptions *opt,
        DataLoader *_data_loader,
        DataLoader *_query_data_loader,
        hnswlib::SpaceInterface<dist_t> *_space,
        string dist_t_type,
        Config *_config): M(16), ef_construction(200) {

        data_dir = opt->dataDir;
        data_path = opt->point_data_path;
        query_data_path = opt->query_data_path;
        max_elements = opt->maxElements;

        data_loader = _data_loader;
        query_data_loader = _query_data_loader;
        space = _space;
        config = _config;

        GroundTruth::calc_gt(data_dir, data_loader, query_data_loader, *space, 0);
        gt_loader = new GroundTruth::GT_Loader(data_dir, data_loader, query_data_loader);

        alg_hnsw = new hnswlib::HierarchicalNSW<dist_t>(space, max_elements, M, ef_construction);
        alg_hnsw->config = config;
        for (int i = 0; i < data_loader->get_elements(); i++) {
            alg_hnsw->addPoint(data_loader->point_data(i), i);
        }

        cout << "build graph finished\n";

        // data_loader->free_data();
    }

    ~Tester() {
        if (data_loader != nullptr)
            delete data_loader;
        delete query_data_loader;
        delete gt_loader;
        delete alg_hnsw;
        delete space;
    }

    void test() {
        test_vs_recall(data_dir, data_loader, query_data_loader, gt_loader, alg_hnsw, 10);
    }

    void test_waste_cands();

    void test_used_neighbor_dist();

    void test_dir_vector();

    vector<dir_vector::Dir_Vector*> dir_vectors;

private:
    string data_dir;
    string data_path;
    string query_data_path;
    DATALOADER::DataLoader *data_loader;
    DATALOADER::DataLoader *query_data_loader;
    GroundTruth::GT_Loader *gt_loader;
    hnswlib::HierarchicalNSW<dist_t> *alg_hnsw;
    hnswlib::SpaceInterface<dist_t> *space;
    Config *config;
    int M;
    int ef_construction = 200;
    int max_elements;
};

template<typename dist_t>
void Tester<dist_t>::test_waste_cands() {

    config->clear_cand();
    config->statis_wasted_cand = 1;
    cout << "ef\t tot cands\t waste cands\t waste/cands\t calculated nodes\t cands/calculated\n";
    for (int ef = 10; ef <= 100; ef += 10) {
        config->clear_cand();
        alg_hnsw->setEf(ef);
        float recall = test_approx(query_data_loader, gt_loader, alg_hnsw, 10);

        cout << ef << "\t" << config->tot_cand_nodes << "\t" << config->wasted_cand_nodes << "\t" << 1.0 * config->wasted_cand_nodes / config->tot_cand_nodes 
            << config->tot_calculated_nodes << "\t" << 1.0 * config->tot_cand_nodes / config->tot_calculated_nodes << '\n';
    }
}

template<typename dist_t>
vector<int> get_hnsw_layer0_neighbors(hnswlib::HierarchicalNSW<dist_t> *hnsw, int id) {
    int *data = (int *) hnsw->get_linklist0(id);
    vector<int> neighbors;
    size_t size = hnsw->getListCount((unsigned int*)data);
    for (int i = 1; i <= size; i++) {
        neighbors.push_back(*(data + i));
    }
    return neighbors;
}
template<typename dist_t>
void Tester<dist_t>::test_used_neighbor_dist() {

    config->statis_used_neighbor_dist = 1;
    size_t qsize = query_data_loader->get_elements();
    alg_hnsw->setEf(30);

    for (int i = 0; i < qsize; i+=10) {
        config->clear_used_neighbors();
        auto ans = alg_hnsw->searchKnn(query_data_loader->point_data(i), 10);
        // cout << "qid = " << i << '\n';

        int nearest_neighbor = ans.top().second;

        int path_len = 0;
        for (auto id: config->used_points) {
            path_len ++ ;
            if (id == nearest_neighbor) break;
        }
        cout << path_len << '\n';
    }
}

template<typename dist_t>
void Tester<dist_t>::test_dir_vector() {

    using dir_vector::Dir_Vector;
    Dir_Vector::init(data_loader->get_dim());
    config->test_dir_vector = 1;


    dir_vectors.resize(data_loader->get_elements());

    for (int i = 0; i < data_loader->get_elements(); i++) {
        auto neighbors = get_hnsw_layer0_neighbors(alg_hnsw, i);
        dir_vectors[i] = new Dir_Vector(neighbors.size());

        if (is_same<dist_t, int>::value) {
            int tot = 0;
            for (auto n: neighbors) {
                dir_vectors[i]->calc_dir_vector_int8(data_loader->point_data(i),
                    data_loader->point_data(n), tot);
                tot++;
            }
        } else if (is_same<dist_t, float>::value) {
            int tot = 0;
            for (auto n: neighbors) {
                dir_vectors[i]->calc_dir_vector_float(data_loader->point_data(i),
                    data_loader->point_data(n), tot);
                tot++;
            }
        }
    }

    // auto neighbors = get_hnsw_layer0_neighbors(alg_hnsw, 1);
    // for (int i = 0; i < neighbors.size(); i++) {
    //     data_loader->print_point_data_int8(neighbors[i]);
    //     dir_vectors[1]->print_dir_vector(i);
    // }

}