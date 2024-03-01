#include <bits/stdc++.h>
#include "data_loader.h"

#include "hnswlib.h"
#include "calc_group_truth.h"
#include "recall_test.h"
#include "ArgParser.h"
#include "config.h"
#include "statis_tasks.h"
#include "dir_vector.h"

using namespace std;
using DATALOADER::DataLoader;



template<typename dist_t>
void begin_tst(Tester<dist_t> *rt, Config *config) {
    // rt->test();
    // rt->test_waste_cands();
    // rt->test_used_neighbor_dist();
    rt->test_dir_vector();
    delete rt;
}

int main(int argc, char *argv[]) {

    CommandLineOptions opt = ArgParser(argc, argv);
    int max_elements = opt.maxElements;

    int M = 16;
    int ef_construction = 200;

    DataLoader *data_loader;
    DataLoader *query_data_loader;

    GroundTruth::GT_Loader *gt_loader;

    Config *config = new Config();

    if (opt.dataName == "bigann") {
        data_loader = new DataLoader("u8", opt.maxElements, opt.point_data_path);
        query_data_loader = new DataLoader("u8", 0, opt.query_data_path);
        hnswlib::SpaceInterface<int> *space = new hnswlib::L2SpaceI(data_loader->get_dim());
        auto *rt = new Tester<int>(&opt, data_loader, query_data_loader, space, "u8", config);

        begin_tst(rt, config);
    } else if (opt.dataName == "yandex") {
        data_loader = new DataLoader("f", opt.maxElements, opt.point_data_path);
        query_data_loader = new DataLoader("f", 0, opt.query_data_path);
        hnswlib::SpaceInterface<float> *space = new hnswlib::L2Space(data_loader->get_dim());
        auto *rt = new Tester<float>(&opt, data_loader, query_data_loader, space, "f", config);

        begin_tst(rt, config);
    }

    return 0;
}