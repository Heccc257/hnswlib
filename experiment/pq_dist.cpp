#include "pq_dist.h"
#include <cstring>
#include <memory>
#include <immintrin.h>

using namespace std;

PQDist::PQDist(int _d, int _m, int _nbits) :d(_d), m(_m), nbits(_nbits) {
    indexPQ = std::move(std::make_unique<faiss::IndexPQ>(d, m, nbits));
    code_nums = 1 << nbits;
    d_pq = _d / _m;
    if (nbits > 8) {
        cout << "Warning nbits exceeds 8: " << nbits << "\n";
    } else if (8 % nbits != 0) {
        perror("nbits must be divided by 8!");
    }

    pq_dist_cache.resize(m * code_nums);
    qdata.resize(d);

    space = std::move(unique_ptr<hnswlib::SpaceInterface<float>> (new hnswlib::L2Space(d_pq)));

}

void PQDist::train(int N, std::vector<float> &xb) {
    indexPQ->train(N, xb.data());
    std::cout << "code size = " << indexPQ->code_size << "\n";
    codes.resize(N * indexPQ->code_size);
    indexPQ->sa_encode(N, xb.data(), codes.data());

    centroids.assign(indexPQ->pq.centroids.begin(), indexPQ->pq.centroids.end());
}

// 获取每个quantizer对应的质心id
vector<int> PQDist::get_centroids_id(int id) {
    const uint8_t *code = codes.data() + id * (this->m * this->nbits / 8);
    vector<int> centroids_id(m);
    int mask = (1<<nbits) - 1;
    int off = 0;
    for (int i = 0; i < m; i++) {
        centroids_id[i] = ((int)(((*code)>>off) & mask));
        off = (off + nbits) & 7; // mod 8
        if (!off) {
            code += 1; // 下一个code字节
        }
    }
    return centroids_id;
}

float* PQDist::get_centroid_data(int quantizer, int code_id) {
    // return indexPQ->pq.centroids.data() + (quantizer*code_nums + code_id) * d_pq;
    return centroids.data() + (quantizer*code_nums + code_id) * d_pq;
}

float PQDist::calc_dist(int d, float *vec1, float *vec2) {
    assert(d == *reinterpret_cast<int*>(space->get_dist_func_param()));
    return space->get_dist_func()(vec1, vec2, space->get_dist_func_param());
    // float ans = 0;
    // for (int i = 0; i < d; i++)
    //     ans += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);f
    // return ans;
}

float PQDist::calc_dist_pq(int data_id, float *qdata, bool use_cache=true) {
    static const float eps = 1e-7;
    float dist = 0;
    auto ids = get_centroids_id(data_id);
    for (int q = 0; q < m; q++) {
        float d;
        if (!use_cache || pq_dist_cache[q*code_nums + ids[q]] < eps) {
            // quantizers
            float *centroid_data = get_centroid_data(q, ids[q]);
            d = calc_dist(d_pq, centroid_data, qdata + (q * d_pq));

            if (use_cache) pq_dist_cache[q*code_nums + ids[q]] = d;
        } else {
            d = pq_dist_cache[q*code_nums + ids[q]];
        }
        dist += d;
    }
    return dist;
}

void PQDist::clear_pq_dist_cache() {
    memset(pq_dist_cache.data(), 0, pq_dist_cache.size() * sizeof(float));
}

void PQDist::load_query_data(const float *_qdata, bool _use_cache) {
    memcpy(qdata.data(), _qdata, sizeof(float) * d);
    clear_pq_dist_cache();
    use_cache = _use_cache;
}
void PQDist::load_query_data_and_cache(const float *_qdata) {
    memcpy(qdata.data(), _qdata, sizeof(float) * d);
    clear_pq_dist_cache();
    use_cache = true;
    for(int i = 0; i < m * code_nums; i++) {
        pq_dist_cache[i] = calc_dist(d_pq, get_centroid_data(i / code_nums, i % code_nums), qdata.data() + (i / code_nums) * d_pq);
    }
}
float PQDist::calc_dist_pq_(int data_id, float *qdata, bool use_cache=true) {
    float dist = 0;
    auto ids = get_centroids_id(data_id);
    for (int q = 0; q < m; q++) {
        dist += pq_dist_cache[q*code_nums + ids[q]];
    }
    return dist;
}
float PQDist::calc_dist_pq_simd(int data_id, float *qdata, bool use_cache) {
    float dist = 0;
    std::vector<int> ids = get_centroids_id(data_id);

    __m256 simd_dist = _mm256_setzero_ps();
    int q;
    for (q = 0; q <= m - 8; q += 8) {
        __m256i id_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ids.data() + q));
        __m256i offset_vec = _mm256_setr_epi32(
            0 * code_nums, 1 * code_nums, 2 * code_nums, 3 * code_nums,
            4 * code_nums, 5 * code_nums, 6 * code_nums, 7 * code_nums
        );
        id_vec = _mm256_add_epi32(id_vec, offset_vec);

        __m256 dist_vec = _mm256_i32gather_ps(pq_dist_cache.data() + q * code_nums, id_vec, 4);

        simd_dist = _mm256_add_ps(simd_dist, dist_vec);
    }

    float dist_array[8];
    _mm256_storeu_ps(dist_array, simd_dist);
    for (int i = 0; i < 8; ++i) {
        dist += dist_array[i];
    }

    for (; q < m; q++) {
        dist += pq_dist_cache[q * code_nums + ids[q]];
    }

    return dist;
}

float PQDist::calc_dist_pq_loaded(int data_id) {
    return calc_dist_pq(data_id, qdata.data(), use_cache);
}
float PQDist::calc_dist_pq_loaded_(int data_id) {
    return calc_dist_pq_(data_id, qdata.data(), use_cache);
}
float PQDist::calc_dist_pq_loaded_simd(int data_id) {
    return calc_dist_pq_simd(data_id, qdata.data(), use_cache);
}

void PQDist::load(string filename) {
    //
    ifstream fin(filename, std::ios::binary);
    if (fin.is_open() == false) {
        cout << "open " << filename << " fail\n";
        exit(-1);
    }
    // GIST
    // n d m nbit
    // int [n * m]
    // float [2^nbits * d]
    int N;
    fin.read(reinterpret_cast<char*>(&N), 4);
    fin.read(reinterpret_cast<char*>(&d), 4);
    fin.read(reinterpret_cast<char*>(&m), 4);
    fin.read(reinterpret_cast<char*>(&nbits), 4);
    cout << "load: " << N << ' ' << d << " " << m << " " << nbits << endl;
    assert(8 % nbits == 0);
    code_nums = 1 << nbits;

    d_pq = d / m;
    space = std::move(unique_ptr<hnswlib::SpaceInterface<float>> (new hnswlib::L2Space(d_pq)));

    pq_dist_cache.resize(m * code_nums);

    codes.resize(N / 8 * m * nbits);
    
    fin.read(reinterpret_cast<char*>(codes.data()), codes.size());

    // cout << "codes " << codes.size() << "\n";
    // for (int i = 0; i < m/2; i++) {
    //     cout << (int)codes[i] << ' ';
    // }
    // cout << "\n";

    centroids.resize(code_nums * d);
    fin.read(reinterpret_cast<char*>(centroids.data()), 4 * centroids.size());

    // auto code = get_centroids_id(0);
    // for (int i = 0; i < m; i++) {
    //     cout << (int)code[i] << ' ';
    // }
    // cout << "\n";
    // exit(0);

    fin.close();
}

vector<int> PQDist::encode_query(float *query) {
    // return indexPQ->pq.centroids.data() + (quantizer*code_nums + code_id) * d_pq;
    vector<int> res;
    for (int q = 0; q < m; q++) {
        int min_id = 0;
        float min_dist = 1e9;
        for (int i = 0; i < code_nums; i++) {
            float d = calc_dist(d_pq, get_centroid_data(q, i), query + (q * d_pq));
            if (d < min_dist) {
                min_dist = d;
                min_id = i;
            }
        }
        res.push_back(min_id);
    }
    return res;
}
void PQDist::construct_distance_table() {
    distance_table.resize(m);
    for (int i = 0; i < m; i++) {
        distance_table[i].resize(1 << nbits);
        for (int j = 0; j < 1 << nbits; j++) {
            distance_table[i][j].resize(1 << nbits);
            for (int k = 0; k < 1 << nbits; k++) {
                distance_table[i][j][k] = calc_dist(d_pq, get_centroid_data(i, j), get_centroid_data(i, k));
            }
        }
    }
}
float PQDist::calc_dist_pq_from_table(int data_id, vector<int>& qids) {
    float dist = 0;
    auto ids = get_centroids_id(data_id);
    for (int q = 0; q < m; q++) {
        dist += distance_table[q][ids[q]][qids[q]];
    }
    return dist;
}
