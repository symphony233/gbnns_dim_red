
#include <chrono>
#include<set>
#include "support_func.h"

using namespace std;


class StopW {
    std::chrono::steady_clock::time_point time_begin;
public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }
};


class KLgraph {
public:
    int L;
    vector< vector <uint32_t> > longmatrixNN;
    void BuildByNumber(int l, vector<float> dataset, size_t N, size_t d, std::mt19937 random_gen,
                       Metric *metric);
    void BuildByNumberCustom(int l, vector<float> dataset, size_t N, size_t d, size_t sqrtN, std::mt19937 random_gen,
                             Metric *metric);
    void BuildByDist(int l, vector<float> dataset, size_t N, size_t d, std::mt19937 random_gen,
                     Metric *metric);
};


void KLgraph::BuildByNumber(int l, vector<float> dataset, size_t N, size_t d, std::mt19937 random_gen,
                    Metric *metric){
    L = l;
    vector<uint32_t> sloy;
    for (int i=0; i < N; ++i) {
        longmatrixNN.push_back(sloy);
    }
    vector<float> custom_prob;
    for (int i=0; i < N - 1; ++i) {
        custom_prob.push_back(1. / (i+ 1) );
    }

    discrete_distribution<int> custom_distr (custom_prob.begin(), custom_prob.end());

    #pragma omp parallel for
    for(int i=0; i < N; ++i) {
        int num;
        const float *point_i = dataset.data() + i*d;
        vector<Neighbor> chosen_neigs;
        set<Neighbor> chn_neigs;
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                const float *point_j = dataset.data() + j * d;
                float dist = metric->Dist(point_i, point_j, d);
                Neighbor neig{j, dist};
                chosen_neigs.push_back(neig);
            }
        }
        sort(chosen_neigs.begin(), chosen_neigs.end());
        unordered_set <int> ll;

        while (ll.size() < L) {
            num = custom_distr(random_gen);
            ll.insert(num);
        }
        for (auto el : ll) {
            longmatrixNN[i].push_back(chosen_neigs[el].number);
        }
    }
}


void KLgraph::BuildByNumberCustom(int l, vector<float> dataset, size_t N, size_t d, size_t sqrtN, std::mt19937 random_gen,
                                  Metric *metric){
    cout << sqrtN << ' ' << N << endl;
    L = l;
    vector<uint32_t> sloy;
    for (int i=0; i < N; ++i) {
        longmatrixNN.push_back(sloy);
    }
    vector<float> custom_prob;
    for (int i=0; i < sqrtN; ++i) {
        custom_prob.push_back(1. / (i+ 1) );
    }

    discrete_distribution<int> custom_distr (custom_prob.begin(), custom_prob.end());
    uniform_int_distribution<int> uniform_distr(0, N - 1);

    #pragma omp parallel for
    for(int i=0; i < N; ++i) {
        int num;
        const float *point_i = dataset.data() + i * d;
        vector<Neighbor> chosen_neigs;
        set<Neighbor> chn_neigs;

        while (chn_neigs.size() < sqrtN) {
            num = uniform_distr(random_gen);

            if (num != i) {
                const float *point_num = dataset.data() + num * d;
                float dist = metric->Dist(point_i, point_num, d);
                Neighbor neig{num, dist};
                chn_neigs.insert(neig);
            }
        }

        for (auto el : chn_neigs) {
            chosen_neigs.push_back(el);
        }

        sort(chosen_neigs.begin(), chosen_neigs.end());
        unordered_set <int> ll;


        while (ll.size() < L) {
            num = custom_distr(random_gen);
            ll.insert(num);
        }

        for (auto el : ll) {
            longmatrixNN[i].push_back(chosen_neigs[el].number);
        }
    }
}


void KLgraph::BuildByDist(int l, vector<float> dataset, size_t N, size_t d, std::mt19937 random_gen,
                    Metric *metric){
    L = l;
    float thr = 0.03;
    vector<uint32_t> sloy;
    for (int i=0; i < N; ++i) {
        longmatrixNN.push_back(sloy);
    }

#pragma omp parallel for
    for(int i=0; i < N; ++i) {
        int num;
        const float *point_i = dataset.data() + i*d;
        vector<Neighbor> chosen_neigs;
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                const float *point_j = dataset.data() + j * d;
                float dist = metric->Dist(point_i, point_j, d);
                if (dist > thr) {
                    Neighbor neig{j, dist};
                    chosen_neigs.push_back(neig);
                }
            }
        }
        unordered_set <int> ll;
        vector<float> custom_prob;
        for (int j = 0; j < chosen_neigs.size(); ++j) {
            float dist_cur = chosen_neigs[j].dist;
            custom_prob.push_back(pow(pow(dist_cur, -1), d));
        }
        discrete_distribution<int> custom_distr (custom_prob.begin(), custom_prob.end());
        while (ll.size() < L) {
            num = custom_distr(random_gen);
            ll.insert(num);
        }
        for (auto el : ll) {
            longmatrixNN[i].push_back(chosen_neigs[el].number);
        }
    }
}