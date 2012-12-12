#ifndef __LABFILE_H__
#define __LABFILE_H__

class Labfile { /*{{{*/
  public:
    Labfile() { Init(); }
    Labfile(string filename) { LoadFile(filename); }
    void Init();
    void condense();
    void push_back(int s, int e, int c, float f = float_inf);
    void CopyLab( const Labfile &ref, const int start, const int end);
    void LoadFile(string filename);
    void SaveLab(string filename) {/*{{{*/
      ofstream fs(filename.c_str());
      if (fs.fail()) {
        ErrorExit(__FILE__, __LINE__, -1, "unable to open file %s\n",
                  filename.c_str());
      }
      SaveLab(fs);
    }/*}}}*/
    void SaveLab(ostream &fs) const;
    void Reverse();
    void DumpData(int start = 0, int last = -1) const {/*{{{*/
      if (start < 0 || start >= num_lab)
        start = 0;
      if (last < 0 || last >= num_lab)
        last = num_lab - 1;

      cout << "====== Labfile.DumpData() ======\n";
      for(int i = start; i <= last; i++) {
        cout << i << ": "
          << start_f[i] << " "
          << end_f[i] << ' '
          << cluster[i] << ' ';
        if (!score.empty()) cout << score[i];
        cout << endl;
      }
      cout << "num_lab: " << last - start + 1 << endl;
    }/*}}}*/
    void DumpData(vector<int> &seg_head_state,
                  vector<int> &seg_tail_state) const {/*{{{*/
      assert(seg_head_state.size() == seg_tail_state.size());
      cout << "====== Labfile ======\n";
      for(unsigned s = 0; s < seg_head_state.size(); s++) {
        for(int i = seg_head_state[s]; i <= seg_tail_state[s]; i++) {
          cout << "S" << i << "\ts" << start_f[i] << "\te" << end_f[i] << '\t' << cluster[i] << "\tSeg" << s << endl;
        }
      }
      cout << "num_lab: " << num_lab << endl;
    }/*}}}*/
    void parseStateSeq(vector<int> &state_seq,
                       vector<float>* likelihood_seq = NULL);
    void parseStateSeq( vector<int> &state_seq, vector<int> &ref_end_f );
    void expandMaxClust(unsigned *p_max_clust) {/*{{{*/
      for(unsigned i = 0; i < cluster.size(); i++)
        if(static_cast<unsigned>(cluster[i]) >= *p_max_clust)
          *p_max_clust = cluster[i]+1;
    }/*}}}*/
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
    float getScore(int t) const { return score[t]; }
    vector<int> *getpCluster() { return &cluster; }
    vector<int> *getpStartf() { return &start_f; }
    vector<int> *getpEndf() { return &end_f; }
  private:
    int numlab_;
    string fname_;
    vector<int> startf_;
    vector<int> endf_;
    vector<string> label_;
    vector<float> score_;
};/*}}}*/

#endif /* __LABFILE_H__ */
