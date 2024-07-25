#include <bits/stdc++.h>
#include "data_loader.h"

#include <faiss/IndexPQ.h>
#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include "../hnswlib/hnswlib.h"
#include "calc_group_truth.h"
#include "recall_test.h"
#include "ArgParser.h"
#include "config.h"
#include "statis_tasks.h"
#include "dir_vector.h"
#include "k_means.h"
#include "ivf_hnsw.h"
#include "pq_dist.h"
#include "timer.h"
#include <random>

using namespace std;
using DATALOADER::DataLoader;
using namespace faiss;

int main(int argc, char *argv[])
{
    // CommandLineOptions opt = ArgParser(argc, argv);
    // int max_elements = opt.maxElements;
    if (argc < 4)
    {
        cout << "Usage: ./val_PQ m nbits use_quantization" << endl;
        return 0;
    }
    int m = atoi(argv[1]);
    int nbits = atoi(argv[2]);
    int use_quantization = atoi(argv[3]);
    int quantization_bits;
    if(use_quantization && argc < 5){
        cout << "Usage: ./val_PQ m nbits use_quantization quantization_bits" << endl;
        return 0;
    }
    if(use_quantization) quantization_bits = atoi(argv[4]);
    int M = 16;
    int ef_construction = 200;
    DataLoader *data_loader = new DataLoader("f", 1000000, "../../gist/train.fvecs", "gist");
    DataLoader *query_data_loader = new DataLoader("f", 1000, "../../gist/test.fvecs", "gist");
    srand(static_cast<unsigned int>(time(0)));
    int d = 960;
    cout << "data prepared" << endl;
    hnswlib::L2Space space(960);
    PQDist *pq_dist;
    if (use_quantization)
    {
        switch (quantization_bits)
        {
        case 8:
            cout << "quantization_bits = 8" << endl;
            pq_dist = new PQDist(d, m, nbits, QuantizationType::UINT8);
            break;
        case 16:
            cout << "quantization_bits = 16" << endl;
            pq_dist = new PQDist(d, m, nbits, QuantizationType::UINT16);
            break;
        case 32:
            cout << "quantization_bits = 32" << endl;
            pq_dist = new PQDist(d, m, nbits, QuantizationType::UINT32);
            break;
        }
    }
    else
    {
        cout << "no quantization" << endl;
        pq_dist = new PQDist(d, m, nbits);
    }
    string path = "../../python_gist/encoded_data_" + to_string(m) + "_" + to_string(nbits);
    pq_dist->load(path);
    ifstream file("/share/ann_benchmarks/point_search.txt");
    string line;
    vector<vector<int>> points_search(1000);
    int t = 0;
    while (getline(file, line))
    {
        istringstream ls(line);
        int id;
        while (ls >> id)
        {
            points_search[t].push_back(id);
        }
        t++;
    }
    StopW LoadTimer;
    LoadTimer.reset();
    if (use_quantization)
    {
        for (int j = 0; j < query_data_loader->get_elements(); j += 1)
        {
            pq_dist->load_query_data_and_cache_quantized(reinterpret_cast<const float *>(query_data_loader->point_data(j)));
        }
    }
    else
    {
        for (int j = 0; j < query_data_loader->get_elements(); j += 1)
        {
            pq_dist->load_query_data_and_cache(reinterpret_cast<const float *>(query_data_loader->point_data(j)));
        }
    }
    cout << "load time " << LoadTimer.getElapsedTimeMicro() / 1e6 << endl;

    StopW PQTimer;
    PQTimer.reset();
    vector<float> pqs;
    if (use_quantization)
    {
        for (int j = 0; j < query_data_loader->get_elements(); j += 1)
        {
            pq_dist->load_query_data_and_cache_quantized(reinterpret_cast<const float *>(query_data_loader->point_data(j)));
            for (int i : points_search[j])
            {
                float distPQ = pq_dist->calc_dist_pq_loaded_simd_quantized(i);
                pqs.push_back(distPQ);
            }
        }
    }
    else
    {
        for (int j = 0; j < query_data_loader->get_elements(); j += 1)
        {
            pq_dist->load_query_data_and_cache(reinterpret_cast<const float *>(query_data_loader->point_data(j)));
            for (int i : points_search[j])
            {
                float distPQ = pq_dist->calc_dist_pq_loaded_simd(i);
                pqs.push_back(distPQ);
            }
        }
    }
    cout << "PQ time " << PQTimer.getElapsedTimeMicro() / 1e6 << endl;
    StopW RealTimer;
    RealTimer.reset();
    vector<float> reals;
    for (int j = 0; j < query_data_loader->get_elements(); j += 1)
    {
        pq_dist->load_query_data(reinterpret_cast<const float *>(query_data_loader->point_data(j)), 1);
        for (int i : points_search[j])
        {
            float real_dist = space.get_dist_func()(
                query_data_loader->point_data(j), data_loader->point_data(i), space.get_dist_func_param());
            reals.push_back(real_dist);
            // cout << distPQ << " " << real_dist << "\n";
        }
    }
    cout << "real time " << RealTimer.getElapsedTimeMicro() / 1e6 << endl;
    assert(pqs.size() == reals.size());

    float error = 0;

    // cout << "\n\n";

    for (int i = 0; i < pqs.size(); i++)
    {
        error += (sqrt(pqs[i]) - sqrt(reals[i])) * (sqrt(pqs[i]) - sqrt(reals[i]));
        // cout << pqs[i] << ' ' << reals[i] << "\n";
    }
    error /= pqs.size();
    cout << "error = " << error << "\n";

    return 0;
}