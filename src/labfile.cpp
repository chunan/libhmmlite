#include <fstream>
#include "labfile.h"
#include "utility.h"

using std::ofstream;
using std::ifstream;
using std::cout;

void Labfile::SaveLab(string filename) const {/*{{{*/
  ofstream fs(filename.c_str());
  if (fs.fail()) {
    ErrorExit(__FILE__, __LINE__, -1, "unable to open file %s\n",
              filename.c_str());
  }
  SaveLab(fs);
}/*}}}*/

void Labfile::LoadFile(string filename) {/*{{{*/
  s_fname = filename;
  int s_val, e_val, c_val;
  num_lab = 0;
  ifstream fs(filename.c_str());
  if (fs.fail()) {
    fprintf(stderr, "Unable to open file %s with flag %s\n", filename.c_str(), "r");
    exit(-1);
  }
  char line_buf[1024];
  while (1) {
    fs.getline(line_buf, 1024);
    if (fs.eof() || fs.fail()) break;
    char *tok;
    if((tok = strtok(line_buf, " ")) == NULL) { // start time
      ErrorExit(__FILE__, __LINE__, 1,
                "label file break at line %d\n", num_lab);
    }
    s_val = atoi(tok);
    if ((tok = strtok(NULL, " ")) == NULL) { // end time
      ErrorExit(__FILE__, __LINE__, 1,
                "label file break at line %d\n", num_lab);
    }
    e_val = atoi(tok);
    assert(e_val >= s_val);
    if ((tok = strtok(NULL, " \n")) == NULL) { // cluster index
      ErrorExit(__FILE__, __LINE__, 1,
                "label file break at line %d\n", num_lab);
    }
    c_val = atoi(tok);

    // push back
    num_lab++;
    start_f.push_back(s_val);
    end_f.push_back(e_val);
    cluster.push_back(c_val);

    if((tok = strtok(NULL, " \n")) != NULL) { // score
      score.push_back(atof(tok));
    }
    if (!score.empty() && score.size() != start_f.size()) {
      ErrorExit(__FILE__, __LINE__, 1,
                "Inconsistent score entries at line %d\n", num_lab);
    }
  }
  fs.close();
}/*}}}*/

void Labfile::SaveLab(ostream &fs) const {/*{{{*/
  assert(start_f.size() == static_cast<unsigned>(num_lab));
  assert(end_f.size() == static_cast<unsigned>(num_lab));
  assert(cluster.size() == static_cast<unsigned>(num_lab));

  for (int i = 0; i < num_lab; ++i) {
    fs << start_f[i] << ' ' << end_f[i] << ' ' << cluster[i];
    if (!score.empty()) fs << ' ' << score[i];
    fs << endl;
  }
}/*}}}*/

void Labfile::Reverse() {/*{{{*/
  if (num_lab > 0) {
    std::reverse(start_f.begin(), start_f.end());
    std::reverse(end_f.begin(), end_f.end());
    std::reverse(cluster.begin(), cluster.end());
    std::reverse(score.begin(), score.end());
  }
}/*}}}*/

void Labfile::Init() {/*{{{*/
  s_fname.clear();
  num_lab = 0;
  start_f.clear();
  end_f.clear();
  cluster.clear();
  score.clear();
}/*}}}*/

void Labfile::push_back(int s, int e, int c, float f) {/*{{{*/
  start_f.push_back(s);
  end_f.push_back(e);
  cluster.push_back(c);
  if (f != float_inf) score.push_back(f);
  num_lab++;
}/*}}}*/

void Labfile::condense() { /*{{{*/
  vector<int>::iterator s_i = start_f.begin() + 1;
  vector<int>::iterator e_i = end_f.begin() + 1;
  vector<int>::iterator c_i = cluster.begin() + 1;
  vector<float>::iterator f_i = score.begin() + 1;
  while (c_i != cluster.end()) {
    if (*c_i == *(c_i - 1)) {
      *(e_i - 1) = *e_i;
      *(f_i - 1) += *f_i;
      s_i = start_f.erase(s_i);
      e_i = end_f.erase(e_i);
      c_i = cluster.erase(c_i);
      f_i = score.erase(f_i);
    }
    else{
      s_i++;
      e_i++;
      c_i++;
      f_i++;
    }
  }
  num_lab = start_f.size();
}/*}}}*/

void Labfile::parseStateSeq(vector<int>& state_seq,/*{{{*/
                            vector<float>* likelihood_seq) {
  assert(&state_seq != &cluster);
  Init();

  if (state_seq.empty()) return;

  float accumLike = 0.0;
  start_f.push_back(0);
  for (unsigned t = 1; t < state_seq.size(); t++)
    if (state_seq[t] != state_seq[t-1]) {
      /* finish this entry */
      end_f.push_back(t-1);
      cluster.push_back(state_seq[t-1]);
      if (likelihood_seq) {
        score.push_back((*likelihood_seq)[t - 1] - accumLike);
        accumLike = (*likelihood_seq)[t - 1];
      }
      /* prepare next entry */
      start_f.push_back(t);
    }
  /* Final entry */
  end_f.push_back(state_seq.size()-1);
  cluster.push_back(state_seq.back());
  if (likelihood_seq)
    score.push_back(likelihood_seq->back() - accumLike);
  num_lab = cluster.size();

} /*}}}*/

void Labfile::parseStateSeq(vector<int> &state_seq, vector<int> &ref_end_f) {/*{{{*/
  assert(&state_seq != &cluster);
  assert(state_seq.size() == static_cast<unsigned>(ref_end_f.back()+1));

  Init();
  for (unsigned s = 0; s < ref_end_f.size(); s++) {
    int s_time = (s == 0) ? 0 : ref_end_f[s-1] + 1;
    int e_time = ref_end_f[s];
    start_f.push_back(s_time);
    for (int t = s_time+1; t <= ref_end_f[s]; t++) {
      if (state_seq[t] != state_seq[t-1]) {
        end_f.push_back(t-1);
        cluster.push_back(state_seq[t-1]);
        start_f.push_back(t);
      }
    }
    end_f.push_back(e_time);
    cluster.push_back(state_seq[e_time]);
  }
  num_lab = cluster.size();

} /*}}}*/

void Labfile::DumpData(int start, int last) const {/*{{{*/
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

void Labfile::DumpData(vector<int> &seg_head_state,/*{{{*/
                       vector<int> &seg_tail_state) const {
  assert(seg_head_state.size() == seg_tail_state.size());
  cout << "====== Labfile ======\n";
  for(unsigned s = 0; s < seg_head_state.size(); s++) {
    for(int i = seg_head_state[s]; i <= seg_tail_state[s]; i++) {
      cout << "S" << i << "\ts" << start_f[i] << "\te" << end_f[i] << '\t' << cluster[i] << "\tSeg" << s << endl;
    }
  }
  cout << "num_lab: " << num_lab << endl;
}/*}}}*/

void Labfile::expandMaxClust(unsigned *p_max_clust) {/*{{{*/
  for(unsigned i = 0; i < cluster.size(); i++)
    if(static_cast<unsigned>(cluster[i]) >= *p_max_clust)
      *p_max_clust = cluster[i]+1;
}/*}}}*/

