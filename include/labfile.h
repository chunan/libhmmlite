#ifndef __LABFILE_H__
#define __LABFILE_H__

#include <vector>
#include <string>
#include <iostream>
#include "logarithmetics.h"

using std::vector;
using std::string;
using std::ostream;

#define float_inf std::numeric_limits<float>::infinity()


class Labfile { /*{{{*/
  public:
    Labfile() { Init(); }
    Labfile(string filename) { LoadFile(filename); }
    void LoadFile(string filename);
    void Init();
    void condense();
    void push_back(int s, int e, int c, float f = float_inf);
    void CopyLab(const Labfile &ref, const int start, const int end);
    void SaveLab(string filename) const;
    void SaveLab(ostream &fs) const;
    void Reverse();
    void DumpData(int start = 0, int last = -1) const;
    void DumpData(vector<int> &seg_head_state, vector<int> &seg_tail_state) const;
    void parseStateSeq(vector<int> &state_seq,
                       vector<float>* likelihood_seq = NULL);
    void parseStateSeq(vector<int> &state_seq, vector<int> &ref_end_f);
    void expandMaxClust(unsigned *p_max_clust);
    void incNumLab() { num_lab++; }
    void setNumLab(int n) { num_lab=n; }
    const char *getFname() const { return s_fname.c_str(); }
    int getNumLab() const { return num_lab; }
    int getStartF(int t) const { return start_f[t]; };
    int getEndF(int t) const { return end_f[t]; };
    int getDuration(int t) const { return end_f[t] - start_f[t] + 1; }
    int getDuration_1(int t) const { return end_f[t] - start_f[t]; }
    int getDuration(int s, int e) const { return end_f[e] - start_f[s] + 1; }
    int getCluster(int t) const {
      return t >= 0 ? cluster[t] : cluster[cluster.size() + t];
    }
    LLDouble getScore(int t) const { return score[t]; }
    vector<int> *getpCluster() { return &cluster; }
    vector<int> *getpStartf() { return &start_f; }
    vector<int> *getpEndf() { return &end_f; }
  private:
    int num_lab;
    string s_fname;
    vector<int> start_f;
    vector<int> end_f;
    vector<int> cluster;
    vector<float> score;
};/*}}}*/

#endif /* __LABFILE_H__ */
