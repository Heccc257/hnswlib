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
    //CommandLineOptions opt = ArgParser(argc, argv);
    //int max_elements = opt.maxElements;
    int m = atoi(argv[1]);
    int nbits = atoi(argv[2]);
    int M = 16;
    int ef_construction = 200;
    DataLoader *data_loader = new DataLoader("f", 1000000, "../../gist/train.fvecs", "gist");
    DataLoader *query_data_loader = new DataLoader("f", 1000, "../../gist/test.fvecs", "gist");
    //GroundTruth::GT_Loader *gt_loader;

    //Config *config = new Config();

    // Set random seed for reproducibility
    srand(static_cast<unsigned int>(time(0)));

    // Dimension of the vectors
    int d = 960;

    // Number of vectors to index
    //int N = opt.maxElements;

    // Number of subquantizers
    //int m = 60;

    // Number of bits per subquantizer
    //int nbits = 4;

    //data_loader = new DataLoader("f", opt.maxElements, opt.point_data_path, "gist");
    //query_data_loader = new DataLoader("f", 0, opt.query_data_path, "gist");
    //gt_loader = new GroundTruth::GT_Loader(opt.dataDir, data_loader, query_data_loader);
    //hnswlib::SpaceInterface<float> *space = new hnswlib::L2Space(data_loader->get_dim());

    cout << "data prepared" << endl;
    hnswlib::L2Space space(960);
    //vector<float> xb(N * d * sizeof(float));
    //for (int i = 0; i < N; i++)
    //{
        //memcpy(xb.data() + (i * d * sizeof(float)), data_loader->point_data(i), d * sizeof(float));
    //}
    //StopW trainTimer;
    PQDist *pq_dist = new PQDist(d, m, nbits);

    string path = "../../python_gist/encoded_data_" + to_string(m) + "_" + to_string(nbits);
    pq_dist->load(path);
    /* for (int i = 99999; i < 100000; i++)
    {
        for (int j = 0; j < 120; j++)
            cout << (int)pq_dist->codes[i * pq_dist->m + j] << " ";
        cout << "\n";
    } */
    // pq_dist->train(N, xb);
    // cout << "train time " << trainTimer.getElapsedTimeMicro() << endl;

    //hnswlib::SpaceInterface<float> *space2 = new hnswlib::L2Space(data_loader->get_dim() / 2);

    ifstream file("../point_search.txt");
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
    //prefetch
    /* for (int j = 0; j < query_data_loader->get_elements(); j += 1)
    {
        pq_dist->load_query_data(reinterpret_cast<const float *>(query_data_loader->point_data(j)), 1);
        for (int i : points_search[j])
        {
            // float distPQ = pq_dist->calc_dist_pq_loaded(i);
            // cout << distPQ << " " << real_dist << "\n";
            float real_dist = space.get_dist_func()(
                query_data_loader->point_data(j), data_loader->point_data(i), space.get_dist_func_param());
        }
    }  */
    StopW LoadTimer;
    LoadTimer.reset();
    for (int j = 0; j < query_data_loader->get_elements(); j += 1)
    {
        pq_dist->load_query_data_and_cache(reinterpret_cast<const float *>(query_data_loader->point_data(j)));
    } 
    cout << "load time " << LoadTimer.getElapsedTimeMicro() / 1e6 << endl;

    StopW PQTimer;
    PQTimer.reset();
    vector<float> pqs;
    for (int j = 0; j < query_data_loader->get_elements(); j += 1)
    {
        pq_dist->load_query_data_and_cache(reinterpret_cast<const float *>(query_data_loader->point_data(j)));
        for (int i : points_search[j])
        {
            float distPQ = pq_dist->calc_dist_pq_loaded_simd(i);
            // cout << distPQ << " " << real_dist << "\n";
            // for (int iter = 0; iter < 2; iter++)
            //     float real_dist = space2->get_dist_func()(
            //             query_data_loader->point_data(j), data_loader->point_data(i), space2->get_dist_func_param());
            pqs.push_back(distPQ);
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

/* int main(int argc, char *argv[])
{
    int m = atoi(argv[1]);
    int nbits = atoi(argv[2]);
    PQDist *pq_dist = new PQDist(960, m, nbits);
    string path = "../../python_gist/encoded_data_" + to_string(m) + "_" + to_string(nbits);
    pq_dist->load(path);
    pq_dist->construct_distance_table();
    DataLoader *data_loader = new DataLoader("f", 1000000, "../../gist/train.fvecs", "gist");
    hnswlib::L2Space space(960);
    DataLoader *query_data_loader = new DataLoader("f", 1000, "../../gist/test.fvecs", "gist");
    ifstream file("../point_search.txt");
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
    float avg_error = 0;
    int times = 0;
    vector<float> true_distances;
    vector<float> pq_distances;
    auto start0 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 1000; i++)
    {
        for(int j : points_search[i])
        {
            float true_distance = space.get_dist_func()(data_loader->point_data(j), query_data_loader->point_data(i), space.get_dist_func_param());
            true_distances.push_back(true_distance);
            times ++;
        }
    }
    auto end0 = std::chrono::high_resolution_clock::now();
    auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(end0 - start0);
    cout << duration0.count() << endl;
    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++)
    {
        float *q = (float *)query_data_loader->point_data(i);
        vector<int> qids = pq_dist->encode_query(q);
        for (int j : points_search[i])
        {

            //cout << true_distance << endl;
            float pq_distance = pq_dist->calc_dist_pq_from_table(j, qids);
            pq_distances.push_back(pq_distance);
            //cout << pq_distance << endl;
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    cout << duration1.count() << endl;
    for (int i = 0; i < times; i++)
    {
        float error = abs(true_distances[i] - pq_distances[i]);
        avg_error += error;
    }
    avg_error /= times;
    cout << avg_error << endl;
} */
/* int step = 1000000;
float avg_error = 0;
int t = 0;
int t_pq = 0;
auto start0 = std::chrono::high_resolution_clock::now();
for(int i = 0; i < 1000; i += step)
for(int j = 0; j < 1000000; j += step)
{
    float true_distance = space.get_dist_func()(data_loader->point_data(j), query_data_loader->point_data(i), space.get_dist_func_param());
    cout << true_distance << endl;
}
auto end0 = std::chrono::high_resolution_clock::now();
auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(end0 - start0);
t = duration0.count();
auto start1 = std::chrono::high_resolution_clock::now();
for(int i = 0; i < 1000; i += step)
{
    float *q = (float *)query_data_loader->point_data(i);
    vector<int> qids = pq_dist->encode_query(q);
    for(int j = 0; j < 1000000; j += step)
    {
        float pq_distance = pq_dist->calc_dist_pq_from_table(j, qids);
        cout << pq_distance << endl;
    }
}
auto end1 = std::chrono::high_resolution_clock::now();
auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
t_pq = duration1.count();
cout << t << " " << t_pq << endl; */
/* for (int i = 0; i < 1000; i += step)
{
    float *q = (float *)query_data_loader->point_data(i);
    vector<int> qids = pq_dist->encode_query(q);
    auto end0 = std::chrono::high_resolution_clock::now();
    auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(end0 - start0);
    t_pq_1 += duration0.count();
    for (int j = 0; j < 100000; j += step)
    {

        float *d = (float *)data_loader->point_data(j);
        auto start1 = std::chrono::high_resolution_clock::now();
        float true_distance = space.get_dist_func()(data_loader->point_data(j), query_data_loader->point_data(i), space.get_dist_func_param());
        //cout << true_distance << endl;
        // float true_distance = 0;
        // for(int i=0;i<960;i++)
        // true_distance += (d[i]-q[i])*(d[i]-q[i]);
        auto end1 = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
        t += duration1.count();
        auto start2 = std::chrono::high_resolution_clock::now();
        float pq_distance = pq_dist->calc_dist_pq_from_table(j, qids);
        //cout << pq_distance << endl;
        auto end2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
        t_pq_2 += duration2.count();
        float error = abs(true_distance - pq_distance);
        avg_error += error;
    }
}
int a = 0;
int b = 0;
auto start = std::chrono::high_resolution_clock::now();
for(int i=0;i<1000;i++)
for(int j=0;j<100000;j++)
{
    auto start = std::chrono::high_resolution_clock::now();
    float dist = space.get_dist_func()(query_data_loader->point_data(i), data_loader->point_data(j), space.get_dist_func_param());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    a += duration.count();
}
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
cout << "time = " << duration.count() << endl;
cout << "time = " << a << endl;

avg_error = avg_error / (d_c * q_c);
cout << t << " " << t_pq_1 << " " << t_pq_2 << " " << avg_error << endl; */
/* } */