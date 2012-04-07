#include "hmmlite.h"
#include "ugoc_utility.h"
#include <cmath>
#include <iomanip>
#include <cstdio>
#include <cstring>
#include <set>
#include <algorithm>

using namespace std;
using namespace atlas;

const int REQUIRED_FILE = 3;


double Gaussian::REQUIRED_FRAME = 30;
double Gaussian::VAR_FLOOR = 0.0;
double GaussianMixture::REQUIRED_FRAME = 40;
double HMM_GMM::TRANS_ZERO = 0.01;
int    HMM_GMM::allowedNDel = 2;


template<class T>
void printVVec(vector<vector< T> >& vvec, const char *msg) {
  printf("\n=================%s=================\n",msg);
  for (unsigned i = vvec.size()-1; i >= 0; i--) {
    for (unsigned t = 0; t < vvec[i].size(); t++) {
      printf("%.2g\t",vvec[i][t]);
    }
    printf("\n");
  }
}

static void G_strip (register char *buf)/*{{{*/
{
  register char *a, *b;

  /* remove leading white space */
  for (a = b = buf; *a == ' ' || *a == '\t'; a++) continue;
  if (a != b) {
    while (*a) {
      *b = *a;
      b++;
      a++;
    }
  }
  /*
     while (*b++ = *a++)
     ;
     */
  /* remove trailing white space */
  for (/*a = buf*/; *a; a++)
    ;
  if (a != buf) {
    for (a--; *a == ' ' || *a == '\t'; a--)
      ;
    a++;
    *a = 0;
  }
}/*}}}*/

static int gettag(FILE *fd, char *tag)/*{{{*/
{
  if (fscanf(fd, "%s", tag) != 1) return 0;
  G_strip (tag);
  //cout << tag << endl;
  return 1;
}/*}}}*/

double LAdd(double x, double y)/*{{{*/
{
  double diff;

  if (x < y) swap(x, y);  // make sure x > y
  diff = y - x;               // diff < 0
  if (diff < MINEARG)
    return (x <= LSMALL) ? LZERO : x ;
  else
    return x + log(1.0 + exp(diff));
}/*}}}*/

double LSub(double x, double y)/*{{{*/
{
  double diff,z;

  if (x < y)
    ErrorExit(__FILE__,__LINE__,-1,"LSub() get negative result\n");
  diff = y - x;
  if (diff < MINEARG)
    return (x <= LSMALL) ? LZERO : x ;
  else {
    z = 1.0 - exp(diff);
    return (z < MINLARG) ? LZERO : x + log(z);
  }
}/*}}}*/

double LDiv(double loga, double logb)/*{{{*/
{
  if (logb <= LSMALL)
    ErrorExit(__FILE__,__LINE__,-1,"LDiv divide by LZERO");
  double z = loga - logb;
  return (z <= LSMALL) ? LZERO : z;
}/*}}}*/

double LProd(double loga, double logb)/*{{{*/
{
  if (loga <= LSMALL || logb <= LSMALL) return LZERO;
  double z = loga + logb;
  return (z <= LSMALL) ? LZERO : z;
}/*}}}*/

double LOG(double a)/*{{{*/
{
  if (a < MINLARG) return LZERO;
  else return log(a);
}/*}}}*/

double EXP(double a)/*{{{*/
{
  if (a < MINEARG) return 0.0;
  else return exp(a);
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

void Labfile::parseStateSeq(vector<int> &state_seq) {/*{{{*/
  assert(&state_seq != &cluster);
  Init();
  start_f.push_back(0);
  for (unsigned t = 1; t < state_seq.size(); t++)
    if (state_seq[t] != state_seq[t-1]) {
      end_f.push_back(t-1);
      cluster.push_back(state_seq[t-1]);
      start_f.push_back(t);
    }
  end_f.push_back(state_seq.size()-1);
  cluster.push_back(state_seq.back());
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



Gaussian::Gaussian(const Gaussian &g)/*{{{*/
{
  Init();
  AllocateMem(g.getDim());
  *p_mean = *g.p_mean;
  *p_cov  = *g.p_cov;
  *p_icov = *g.p_icov;
  logConst = g.logConst;
}/*}}}*/

void Gaussian::Init()/*{{{*/
{
  dim = 0;
  p_mean = 0;
  p_cov = 0;
  p_icov = 0;
  weight = 0.0;
  numframe = 0;
  logConst = 0.0;
  isDiag = false;
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_FAST_NP);
  pthread_mutex_init(&G_mutex, &attr);
}/*}}}*/

void Gaussian::AllocateMem(int d)/*{{{*/
{
  if (d <= 0) return;
  if (d != dim) {
    DeallocateMem();
    dim = d;
    p_mean = new Matrix(d,1);
    p_cov  = new Matrix(d,d);
    p_icov = new Matrix(d,d);
  }
  ClearGaussian();
}/*}}}*/

double Gaussian::InvertCov()/*{{{*/
{
  pthread_mutex_lock(&G_mutex);

  *p_icov = *p_cov;
  double logdet = AisCholeskySymmetricA(p_icov);
  if (logdet != -std::numeric_limits<double>::infinity()) {
    AisInvCholeskyA(p_icov);
    logdet = - logdet / 2;
    logConst = logdet;
  }
  else{
    logdet   = std::numeric_limits<double>::infinity();
  }

  pthread_mutex_unlock(&G_mutex);

  return logdet;
}/*}}}*/

  void Gaussian::AddData/*{{{*/
(const double *data, const int dim, const double prob, UpdateType udtype)
{
  if (!(prob >= 0)) {
    ErrorExit(__FILE__,__LINE__,-1,"Illigal prob (%f) in Gaussian::AddData()\n",prob);
  }

  if (prob < ZERO) return;

  pthread_mutex_lock(&G_mutex);

  numframe++;
  weight += prob;
  for (int r = 0; r < dim; r++) {
    double val = prob * data[r];
    if (udtype == UpdateAll || udtype == UpdateMean)
      p_mean->setEntryPlus(r,0,val);
    if (udtype == UpdateAll || udtype == UpdateCov) {
      p_cov->setEntryPlus(r,r, val * data[r]);
      if (!isDiag) {
        for (int c = r+1; c < dim; c++)
          p_cov->setEntryPlus(r,c, val * data[c]);
      }
    }
  }

  pthread_mutex_unlock(&G_mutex);

}/*}}}*/

void Gaussian::AddMeanCov(const Gaussian &g)/*{{{*/
{

  pthread_mutex_lock(&G_mutex);

  *p_mean += *(g.p_mean);
  p_cov->AddUpper(1.0, *(g.p_cov));

  pthread_mutex_unlock(&G_mutex);
}/*}}}*/

void Gaussian::ClearGaussian()/*{{{*/
{

  pthread_mutex_lock(&G_mutex);

  weight = 0.0;
  numframe = 0;
  p_mean->zeroFill();
  p_cov->zeroFill();

  pthread_mutex_unlock(&G_mutex);

}/*}}}*/

void Gaussian::AddVarFloor()/*{{{*/
{
  if (VAR_FLOOR < ZERO) return;

  pthread_mutex_lock(&G_mutex);

  for (int i = 0; i < dim; i++)
    p_cov->setEntryPlus(i,i,VAR_FLOOR);

  pthread_mutex_unlock(&G_mutex);

}/*}}}*/

bool Gaussian::normMeanCov(const bool subtractmean, UpdateType udtype, double N)/*{{{*/
{

  if (N < 0) {
    N = weight;
    //cerr << "Total weight for Gaussian data = " << N << endl;
    if (numframe <= Gaussian::REQUIRED_FRAME) {
      cerr << "Gaussian data frame (" << numframe << ") not enough. At least "
        << Gaussian::REQUIRED_FRAME << " frames are required.\n";
      return false;
    }
  }
  assert(N > 0);

  pthread_mutex_lock(&G_mutex);

  if (udtype == UpdateAll || udtype == UpdateMean) {
    for (int r = 0; r < dim; r++)
      p_mean->setEntryDiv(r,0,N);
  }

  // If UpdateCov, mean must be set before calling normMeanCov()
  if (udtype == UpdateAll || udtype == UpdateCov) {
    for (int r = 0; r < dim; r++) {
      p_cov->setEntryDiv(r,r,N);
      if (subtractmean) p_cov->setEntryPlus(r,r, -p_mean->entry(r,0)*p_mean->entry(r,0));
      if (!isDiag) {
        for (int c = r+1; c < dim; c++) {
          p_cov->setEntryDiv(r,c,N);
          if (subtractmean) p_cov->setEntryPlus(r,c, -p_mean->entry(r,0)*p_mean->entry(c,0));
        }
      }
    }
  }

  pthread_mutex_unlock(&G_mutex);

  return true;

}/*}}}*/

const Gaussian & Gaussian::operator=(const Gaussian &g)/*{{{*/
{
  pthread_mutex_lock(&G_mutex);

  dim = g.dim;
  *p_mean  = *g.p_mean;
  *p_cov   = *g.p_cov;
  *p_icov  = *g.p_icov;
  logConst = g.logConst;
  isDiag   = g.isDiag;

  pthread_mutex_unlock(&G_mutex);

  return *this;
}/*}}}*/

double Gaussian::Bhat_dist(const Gaussian &g1, const Gaussian &g2) {/*{{{*/
  assert(g1.dim == g2.dim);
  double dist;
  dim = g1.dim;
  /* Get P = (P1 + P2) / 2 */
  *p_cov  = *g1.p_cov;
  *p_cov += *g2.p_cov;
  *p_cov *= 0.5;
  InvertCov();

  *p_mean = *g1.p_mean;
  dist = -0.25 * logProb(g2.p_mean->pointer(), dim, true);
  dist -= 0.75 * logConst;
  dist += 0.5  * (g1.logConst + g2.logConst);
  return dist;
}/*}}}*/

void Gaussian::CopyMean(const Gaussian &g)/*{{{*/
{
  pthread_mutex_lock(&G_mutex);

  assert(dim == g.dim);
  *p_mean  = *(g.p_mean);

  pthread_mutex_unlock(&G_mutex);

}/*}}}*/

void Gaussian::CopyCov(const Gaussian &g)/*{{{*/
{
  pthread_mutex_lock(&G_mutex);

  assert(dim == g.dim);
  *p_cov  = *(g.p_cov);
  *p_icov = *(g.p_icov);
  logConst = g.logConst;

  pthread_mutex_unlock(&G_mutex);

}/*}}}*/

void Gaussian::backoff(const Gaussian &g, double backoff_weight)/*{{{*/
{
  if (backoff_weight < ZERO) return;
  assert(dim = g.dim);
  assert(backoff_weight > 0 && backoff_weight < 1);
  double self_weight = 1 - backoff_weight;
  pthread_mutex_lock(&G_mutex);

  // C = a1 * C1  +  a2 * C2  +  a1 * v1 * v1'  +  a2 * v2 * v2'  -  v * v'

  // p_cov   = a1 * C1 + a2 * C2
  for (int r = 0; r < dim; r++) {
    p_cov->setEntry(r, r, self_weight * p_cov->entry(r,r) + backoff_weight * g.p_cov->entry(r,r));
    if (!isDiag) {
      for (int c = r+1; c < dim; c++)
        p_cov->setEntry(r, c, self_weight * p_cov->entry(r,c) + backoff_weight * g.p_cov->entry(r,c));
    }
  }


  vector<double> mean1, mean2;
  mean1.resize(dim);
  mean2.resize(dim);
  for (int r = 0; r < dim; r++) {
    // mean1 = a1 * v1
    mean1[r] = self_weight * p_mean->entry(r,0) ;
    // mean2 = a2 * v2
    mean2[r] = backoff_weight * g.p_mean->entry(r,0) ;
  }
  for (int r = 0; r < dim; r++) {
    // p_cov = a1 * C1 + a2 * C2 + (a1 * v1 * v1' + a2 * v2 * v2')
    p_cov->setEntryPlus(r,r, mean1[r] * p_mean->entry(r,0) + mean2[r] * g.p_mean->entry(r,0));
    if (!isDiag) {
      for (int c = r+1; c < dim; c++)
        p_cov->setEntryPlus(r,c, mean1[r] * p_mean->entry(c,0) + mean2[r] * g.p_mean->entry(c,0));
    }
  }
  for (int r = 0; r < dim; r++) {
    // p_mean = (a1 * v1) + (a2 * v2)
    p_mean->setEntry(r,0, mean1[r] + mean2[r]);
  }
  for (int r = 0; r < dim; r++) {
    p_cov->setEntryPlus(r,r, -p_mean->entry(r,0) * p_mean->entry(r,0));
    if (!isDiag) {
      for (int c = r+1; c < dim; c++)
        p_cov->setEntryPlus(r,c, -p_mean->entry(r,0) * p_mean->entry(c,0));
    }
  }

  pthread_mutex_unlock(&G_mutex);

}/*}}}*/

void Gaussian::display(FILE *fp) const /*{{{*/
{
  fprintf(fp,"Gaussian ascii\n");

  fprintf(fp,"dim: %d\n",dim);

  fprintf(fp,"mean:");
  for (int r = 0; r < dim; r++)
    fprintf(fp," %g",p_mean->entry(r,0));
  fprintf(fp,"\n");

  fprintf(fp,"cov:\n");
  for (int r = 0; r < dim; r++) {
    for (int c = r; c < dim; c++)
      fprintf(fp," %g",p_cov->entry(r,c));
    fprintf(fp,"\n");
  }


  if (fp == stdout || fp == stderr) {
    cout << "icov:\n";
    for (int r = 0; r < dim; r++) {
      for (int c = r; c < dim; c++)
        cout << fixed << p_icov->entry(r,c) << ' ';
      cout << '\n';
    }
    cout << "logConst = " << logConst << endl;
  }

  fprintf(fp,"EndGaussian\n");

  //#endif
}/*}}}*/

double Gaussian::getTotalVar() const/*{{{*/
{
  double totvar = 0.0;
  for (int d = 0; d < getDim(); d++) {
    totvar += p_cov->entry(d,d);
  }
  return totvar;

}/*}}}*/

void GaussianMixture::Init(int d, int x, vector<Gaussian *> *p)/*{{{*/
{
  dim = d;
  v_weight.resize(x);
  v_gaussidx.resize(x);
  pGaussPool = p;
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_FAST_NP);
  pthread_mutex_init(&S_mutex, &attr);

}/*}}}*/

const GaussianMixture & GaussianMixture::operator=(const GaussianMixture & gm)/*{{{*/
{
  assert(dim == gm.dim);
  assert(v_weight.size() == gm.v_weight.size());

  pthread_mutex_lock(&S_mutex);

  v_weight = gm.v_weight;
  /*
     for (unsigned x = 0; x < v_weight.size(); x++)
   *getpGauss(x) = *gm.getpGauss(x);
   */

  pthread_mutex_unlock(&S_mutex);

  return *this;

}/*}}}*/

void GaussianMixture::setGaussIdx(unsigned x, unsigned idx) { /*{{{*/
  assert(x < v_weight.size() && pGaussPool != 0 && idx < pGaussPool->size());

  pthread_mutex_lock(&S_mutex);

  v_gaussidx[x] = idx;

  pthread_mutex_unlock(&S_mutex);
}/*}}}*/

void GaussianMixture::copyGaussIdx(const GaussianMixture *pstate)  /*{{{*/
{
  pthread_mutex_lock(&S_mutex);

  v_gaussidx = pstate->v_gaussidx;

  pthread_mutex_unlock(&S_mutex);
}/*}}}*/

void GaussianMixture::setWeight(int x, double val, SetType s) /*{{{*/
{
  pthread_mutex_lock(&S_mutex);

  if (s == SET) v_weight[x] = val;
  else if (s == ADD) v_weight[x] += val;
  else fprintf(stderr,"GaussianMixture::setWeight(): Unknown SetType\n");

  pthread_mutex_unlock(&S_mutex);
}/*}}}*/

void GaussianMixture::copyWeight(GaussianMixture &s)/*{{{*/
{
  assert(v_gaussidx.size() <= s.v_gaussidx.size());
  for (unsigned x = 0; x < v_gaussidx.size(); x++)
    setWeight(x, s.getWeight(x),SET);
}/*}}}*/

void GaussianMixture::UniformWeight()/*{{{*/
{
  if (v_weight.empty()) return;
  double w = 1.0 / static_cast<double>(v_weight.size());
  for (unsigned x = 0; x < v_weight.size(); x++)
    v_weight[x] = w;
}/*}}}*/

bool GaussianMixture::containGidx(int gidx)/*{{{*/
{
  for (unsigned x = 0; x < v_gaussidx.size(); x++)
    if (v_gaussidx[x] == gidx) return true;
  return false;
}/*}}}*/

bool GaussianMixture::cancelGaussIdx(int idx)/*{{{*/
{
  pthread_mutex_lock(&S_mutex);

  for (unsigned x = 0; x < v_gaussidx.size(); x++) {
    if (v_gaussidx[x] > idx) v_gaussidx[x]--;
    else if (v_gaussidx[x] == idx) {
      cerr << "Error: v_gaussidx[" << x << "] = " << idx << " is in recycler\n";
      pthread_mutex_unlock(&S_mutex);
      return false;
    }
  }

  pthread_mutex_unlock(&S_mutex);
  return true;
}/*}}}*/

void GaussianMixture::ClearWeight()/*{{{*/
{
  pthread_mutex_lock(&S_mutex);

  for (unsigned x = 0; x < v_weight.size(); x++) v_weight[x] = 0.0;

  pthread_mutex_unlock(&S_mutex);
}/*}}}*/

bool GaussianMixture::normWeight(double & weightsum)/*{{{*/
{

  weightsum = 0.0;
  for (unsigned x = 0; x < v_weight.size(); x++)
    weightsum += v_weight[x];
  if (weightsum < GaussianMixture::getRequiredFrame() * v_weight.size()) {
    cerr << "State frame (" << weightsum << ") too small, not updated.\n";
    return false;
  }

  pthread_mutex_lock(&S_mutex);

  for (unsigned x = 0; x < v_weight.size(); x++)
    v_weight[x] /= weightsum;

  pthread_mutex_unlock(&S_mutex);

  return true;
}/*}}}*/

void GaussianMixture::display(FILE *fp) const/*{{{*/
{
  fprintf(fp, "GaussianMixture ascii\n");

  fprintf(fp, "dim: %d\n", dim);

  fprintf(fp, "nmix: %d\n", static_cast<int>(v_weight.size()));

  fprintf(fp, "weight:");
  for (unsigned x = 0; x < v_weight.size(); x++)
    fprintf(fp, " %g", v_weight[x]);
  fprintf(fp, "\n");

  fprintf(fp, "gaussidx:");
  for (unsigned x = 0; x < v_gaussidx.size(); x++)
    fprintf(fp, " %d", v_gaussidx[x]);
  fprintf(fp, "\n");

  fprintf(fp, "EndGaussianMixture\n");

  if (fp == stdout || fp == stderr)
    cout << "Gaussian pool pointer = " << pGaussPool << endl;
}/*}}}*/

void HMM_GMM::Init()/*{{{*/
{
  i_nstate = 0;
  use = 0;
  pdf_weight = 1.0;
  pStatePool[0] = pStatePool[1] = 0;
  pGaussPool[0] = pGaussPool[1] = 0;
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_FAST_NP);
  pthread_mutex_init(&H_mutex, &attr);
  isLog = false;
}/*}}}*/

void HMM_GMM::CopyForThread(const HMM_GMM &model)/*{{{*/
{
  i_nstate = model.i_nstate;
  use = model.use;
  isLog = model.isLog;
  v_state = model.v_state;
  left = model.left;
  right = model.right;
  for (int u = 0; u < 2; u++) {
    pi[u] = model.pi[u];
    pStatePool[u] = model.pStatePool[u];
    pGaussPool[u] = model.pGaussPool[u];
  }
  v_trans = model.v_trans;
  v_rtrans.resize(i_nstate);
  for (int i = 0; i < i_nstate; i++)
    v_rtrans[i].resize(i_nstate);
  accum_ij = model.accum_ij;
  gauss_isUsed = model.gauss_isUsed;
  state_isUsed = model.state_isUsed;
  pdf_weight = model.pdf_weight;

  /*
     cout << "=============original=============\n";
     model.display(stdout);
     cout << "=============copy cat=============\n";
     display(stdout);
     */

}/*}}}*/

void HMM_GMM::dump_param() const/*{{{*/
{
  cout << "Num. of state = " << i_nstate << endl
    << "Parameter in use = " << use << endl;
  cout << "State index = {";
  for (unsigned i = 0; i < v_state.size(); i++)
    cout << ' ' << v_state[i];
  cout << "}\n";
  for (int u = 0; u < 2; u++) {
    cout << "Pi[" << u << "] = {";
    for (unsigned i = 0; i < pi[u].size(); i++)
      cout << ' ' << pi[u][i];
    cout << "}\n";
  }
  cout << "Trans = {\n";
  for (unsigned i = 0; i < v_trans.size(); i++) {
    for (unsigned j = 0; j < v_trans[i].size(); j++)
      cout << ' ' << setw(5) << v_trans[i][j];
    cout << endl;
  }
  cout << "}\n";
  cout << "RTrans = {\n";
  for (unsigned i = 0; i < v_rtrans.size(); i++) {
    for (unsigned j = 0; j < v_rtrans[i].size(); j++)
      cout << ' ' << setw(5) << v_rtrans[i][j];
    cout << endl;
  }
  cout << "}\n";
  cout << "pStatePool = {" << pStatePool[0] << ", " << pStatePool[1] << "}\n";
  cout << "pGaussPool = {" << pGaussPool[0] << ", " << pGaussPool[1] << "}\n";

}/*}}}*/

void HMM_GMM::display(FILE *fp) const/*{{{*/
{
  fprintf(fp,"HMM ascii\n");

  fprintf(fp,"nstate: %d\n",i_nstate);

  fprintf(fp,"pdf_weight: %g\n",pdf_weight);

  fprintf(fp,"state:");
  for (unsigned i = 0; i < v_state.size(); i++)
    fprintf(fp," %d",v_state[i]);
  fprintf(fp,"\n");

  fprintf(fp,"pi:");
  for (unsigned i = 0; i < pi[use].size(); i++)
    fprintf(fp," %g",pi[use][i]);
  fprintf(fp,"\n");
  if (fp == stdout || fp == stderr) {
    fprintf(fp,"pi:");
    for (unsigned i = 0; i < pi[1-use].size(); i++)
      fprintf(fp," %g",pi[1-use][i]);
    fprintf(fp,"\n");
  }

  fprintf(fp,"left:");
  for (unsigned i = 0; i < left.size(); i++)
    fprintf(fp," %d",left[i]);
  fprintf(fp,"\n");

  fprintf(fp,"right:");
  for (unsigned i = 0; i < right.size(); i++)
    fprintf(fp," %d",right[i]);
  fprintf(fp,"\n");

  fprintf(fp,"trans:\n");
  for (unsigned i = 0; i < v_trans.size(); i++) {
    for (unsigned j = 0; j < v_trans[i].size(); j++)
      fprintf(fp," %g",v_trans[i][j]);
    fprintf(fp,"\n");
  }

  /*
     fprintf(fp,"rtrans:\n");
     for (unsigned i = 0; i < v_rtrans.size(); i++) {
     for (unsigned j = 0; j < v_rtrans[i].size(); j++)
     fprintf(fp," %g",v_rtrans[i][j]);
     fprintf(fp,"\n");
     }
     */


  fprintf(fp,"EndHMM\n");

  if (fp == stdout || fp == stderr) {
    fprintf(fp,"pStatePool = %p\n",pStatePool[use]);
    fprintf(fp,"pGaussPool = %p\n",pGaussPool[use]);
  }

}/*}}}*/

int HMM_GMM::getGMidx(int s) const {/*{{{*/
  assert(s < i_nstate);
  return v_state[s];
}/*}}}*/

GaussianMixture * HMM_GMM::getpGM(int s, UseType u) const /*{{{*/
{
  assert(s < i_nstate);
  int who = (u == USE) ? use : 1-use;
  return pStatePool[who]->operator[](v_state[s]);
}/*}}}*/

double HMM_GMM::getPi(int s, UseType u) const /*{{{*/
{
  assert(s < i_nstate);
  if (u == USE) return pi[use][s];
  else return pi[1-use][s];
}/*}}}*/

double HMM_GMM::getTrans(int i, int j) const /*{{{*/
{
  assert(i<i_nstate && j<i_nstate);
  return v_trans[i][j];
}/*}}}*/

double HMM_GMM::getRTrans(int i, int j) const /*{{{*/
{
  assert(i<i_nstate && j<i_nstate);
  return v_rtrans[i][j];
}/*}}}*/

vector<GaussianMixture *> * HMM_GMM::getpStatePool(UseType u) const /*{{{*/
{
  if (u == USE) return pStatePool[use];
  else return pStatePool[1-use];
}/*}}}*/

vector<Gaussian *> * HMM_GMM::getpGaussPool(UseType u) const /*{{{*/
{
  if (u == USE) return pGaussPool[use];
  else return pGaussPool[1-use];
}/*}}}*/

void HMM_GMM::setNstate(int n)/*{{{*/
{
  i_nstate = n;
  v_state.resize(n);
  left.resize(n);
  right.resize(n);
  v_trans.resize(n);
  v_rtrans.resize(n);
  for (int i = 0; i < n; i++) {
    left[i] = right[i] = -1;
    v_trans[i].resize(n);
    v_rtrans[i].resize(n);
  }
  for (int u = 0; u < 2; u++) {
    pi[u].resize(n);

  }
}/*}}}*/

void HMM_GMM::setState(unsigned s, unsigned idx) { /*{{{*/
  assert(static_cast<int>(s) < i_nstate);
  assert(pStatePool[0] != 0 && idx < pStatePool[0]->size() &&
         pStatePool[1] != 0 && idx < pStatePool[1]->size());
  v_state[s] = idx;
} /*}}}*/

void HMM_GMM::setPi(int s, double val, UseType u) {/*{{{*/
  assert(s < i_nstate);
  pthread_mutex_lock(&H_mutex);
  if (u == USE) pi[use][s] = val;
  else pi[1-use][s] = val;
  pthread_mutex_unlock(&H_mutex);
}/*}}}*/

void HMM_GMM::UniformPi()/*{{{*/
{
  if (i_nstate <= 0) return;
  double share_p = 1.0 / double(i_nstate);
  for (int i = 0; i < i_nstate; i++)
    pi[use][i] = share_p;
}/*}}}*/

void HMM_GMM::addPi(int s, double val, UseType u) {/*{{{*/
  assert(s < i_nstate);
  pthread_mutex_lock(&H_mutex);
  if (u == USE) pi[use][s] += val;
  else pi[1-use][s] += val;
  pthread_mutex_unlock(&H_mutex);
}/*}}}*/

void HMM_GMM::setTrans(int i, int j, double val) { /*{{{*/
  assert(i<i_nstate && j<i_nstate);
  pthread_mutex_lock(&H_mutex);
  v_trans[i][j] = val;
  pthread_mutex_unlock(&H_mutex);
}/*}}}*/

void HMM_GMM::addTrans(int i, int j, double val) { /*{{{*/
  assert(i<i_nstate && j<i_nstate);
  pthread_mutex_lock(&H_mutex);
  v_trans[i][j] += val;
  pthread_mutex_unlock(&H_mutex);
}/*}}}*/

void HMM_GMM::setLeft(int s, int L)/*{{{*/
{
  assert(s >= 0 && s < i_nstate);
  assert(L == -1 || (L >= 0 && L < i_nstate));
  //cout << "HMM_GMM::setLeft(" << s << ", " << L << ")\n";
  left[s] = L;
}/*}}}*/

void HMM_GMM::setRight(int s, int R)/*{{{*/
{
  assert(s >= 0 && s < i_nstate);
  assert(R == -1 || (R >= 0 && R < i_nstate));
  //cout << "HMM_GMM::setRight(" << s << ", " << R << ")\n";
  right[s] = R;
}/*}}}*/

void HMM_GMM::deleteState(int i)/*{{{*/
{
  //cout << "HMM_GMM::deleteState(" << i << ")\n";
  assert(i >= 0 && i < i_nstate);
  /* v_state index */
  v_state.erase(v_state.begin()+i);
  /* left | right */
  left.erase(left.begin()+i);
  right.erase(right.begin()+i);
  for (unsigned j = 0; j < left.size(); j++) {
    /* case: i->j */
    if (left[j] == i) {
      cerr << "Warning: following state s" << j
        << ". Make sure it will be deleted after this operation"
        << endl;
      left[j] = -1;
    }
    else if (left[j] > i) left[j]--;
    /* case: j->i */
    if (right[j] == i) {
      cerr << "Warning: state s" << j << " will have no descendent"
        << endl;
      right[j] = -1;
    }
    else if (right[j] > i) right[j]--;
  }
  /* v_trans */
  v_trans.erase(v_trans.begin()+i);
  for (unsigned j = 0; j < v_trans.size(); j++)
    v_trans[j].erase(v_trans[j].begin()+i);

  /* v_rtrans */
  v_rtrans.erase(v_rtrans.begin()+i);
  for (unsigned j = 0; j < v_rtrans.size(); j++)
    v_rtrans[j].erase(v_rtrans[j].begin()+i);

  for (int u = 0; u < 2; u++) {
    /* v_pi */
    pi[u].erase(pi[u].begin()+i);
  }
  i_nstate--;
}/*}}}*/

bool HMM_GMM::deleteStateIdx(int idx)/*{{{*/
{
  //cout << "HMM_GMM::deleteStateIdx(" << idx << ")\n";
  int i = 0;
  for (; i < i_nstate; i++)
    if (idx == v_state[i]) {
      deleteState(i);
      return true;
    }
  cerr << "Cannot find index " << idx << " in v_state\n";
  return false;
}/*}}}*/

bool HMM_GMM::cancelStateIdx(int idx)/*{{{*/
{
  for (int i = 0; i < i_nstate; i++)
    if (v_state[i] > idx) v_state[i]--;
    else if (v_state[i] == idx) {
      cerr << "Error: v_state[" << i << "] = " << idx << " is in recycler\n";
      return false;
    }
  return true;
}/*}}}*/

int HMM_GMM::insertState(int state_idx)/*{{{*/
{
  assert(state_idx >= 0 && state_idx < static_cast<int>(getpStatePool()->size()));

  int state_no = i_nstate;
  i_nstate++;

  v_state.push_back(state_idx);

  left.push_back(-1);
  right.push_back(-1);

  for (unsigned i = 0; i < v_trans.size(); i++)
    v_trans[i].push_back(0.0);
  v_trans.resize(i_nstate);
  v_trans[state_no].resize(i_nstate);
  for (unsigned i = 0; i < v_trans[state_no].size(); i++)
    v_trans[state_no][i] = 0.0;

  for (unsigned i = 0; i < v_rtrans.size(); i++)
    v_rtrans[i].push_back(0.0);
  v_rtrans.resize(i_nstate);
  v_rtrans[state_no].resize(i_nstate);
  for (unsigned i = 0; i < v_rtrans[state_no].size(); i++)
    v_rtrans[state_no][i] = 0.0;

  for (int u = 0; u < 2; u++) {
    pi[u].push_back(0.0);
  }
  return i_nstate-1;
}/*}}}*/

void HMM_GMM::setpStatePool(vector<GaussianMixture*> *ptr, UseType u) { /*{{{*/
  if (u == USE) pStatePool[use] = ptr;
  else pStatePool[1-use] = ptr;
}/*}}}*/

void HMM_GMM::setpGaussPool(vector<Gaussian*> *ptr, UseType u) { /*{{{*/
  if (u == USE) pGaussPool[use] = ptr;
  else pGaussPool[1-use] = ptr;
}/*}}}*/

void HMM_GMM::ClearPi(UseType u) {/*{{{*/
  int who = (u == USE) ? use : 1-use;
  pthread_mutex_lock(&H_mutex);
  for (unsigned i = 0; i < pi[who].size(); i++) pi[who][i] = 0.0;
  pthread_mutex_unlock(&H_mutex);
}/*}}}*/

void HMM_GMM::ClearTrans() {/*{{{*/
  for (unsigned i = 0; i < v_trans.size(); i++) {
    for (unsigned j = 0; j < v_trans[i].size(); j++)
      v_trans[i][j] = 0.0;
  }
  for (unsigned i = 0; i < v_rtrans.size(); i++) {
    for (unsigned j = 0; j < v_rtrans[i].size(); j++)
      v_rtrans[i][j] = 0.0;
  }
}/*}}}*/

void HMM_GMM::ViteInit()/*{{{*/
{
  LogPi();
  LogTrans();
  isLog = true;
}/*}}}*/

void HMM_GMM::EMInitIter() /*{{{*/
{
  int unuse = 1-use;

  SyncUsed();
  LogPi();
  LogTrans();
  isLog = true;

  /* Pi */
  ClearPi(UNUSE);

  /* Trans */
  ClearIJ();

  /* state->weight */
  for (unsigned i = 0; i < pStatePool[unuse]->size(); i++)
    if (state_isUsed[i]) (*pStatePool[unuse])[i]->ClearWeight();

  /* Gaussian pool */
  vector<Gaussian *> &vGauss = *getpGaussPool(UNUSE);
  for (unsigned g = 0; g < vGauss.size(); g++)
    if (gauss_isUsed[g]) vGauss[g]->ClearGaussian();

  /* Number of utterances*/
  /*
     numutt.resize(vGauss.size());
     for (unsigned g = 0; g < numutt.size(); g++) numutt[g] = 0;
     */

}/*}}}*/

void HMM_GMM::ClearIJ()/*{{{*/
{
  accum_ij.resize(i_nstate);
  for (unsigned i = 0; i < accum_ij.size(); i++) {
    accum_ij[i].resize(i_nstate);
    for (unsigned j = 0; j < accum_ij[i].size(); j++)
      accum_ij[i][j] = 0.0;
  }
}/*}}}*/

void HMM_GMM::LogPi()/*{{{*/
{
  if (isLog) return;
  for (unsigned i = 0; i < pi[use].size(); i++)
    pi[use][i] = LOG(pi[use][i]);
}/*}}}*/

void HMM_GMM::LogTrans()/*{{{*/
{
  if (isLog) return;
  for (unsigned i = 0; i < v_trans.size(); i++)
    for (unsigned j = 0; j < v_trans[i].size(); j++)
      v_trans[i][j] = LOG(v_trans[i][j]);
}/*}}}*/

void HMM_GMM::ExpPxs()/*{{{*/
{
  for (unsigned i = 0; i < px_s.size(); i++)
    for (unsigned x = 0; x < px_s[i].size(); x++)
      for (unsigned t = 0; t < px_s[i][x].size(); t++)
        px_s[i][x][t] = EXP(px_s[i][x][t]);
}/*}}}*/

void HMM_GMM::ExpGamma()/*{{{*/
{
  for (unsigned i = 0; i < gamma.size(); i++)
    for (unsigned t = 0; t < gamma[i].size(); t++)
      gamma[i][t] = EXP(gamma[i][t]);
}/*}}}*/

void HMM_GMM::ExpEpsilon()/*{{{*/
{
  for (unsigned i = 0; i < epsilon.size(); i++)
    for (unsigned j = 0; j < epsilon[i].size(); j++)
      for (unsigned t = 0; t < epsilon[i][j].size(); t++)
        epsilon[i][j][t] = EXP(epsilon[i][j][t]);
}/*}}}*/

#ifdef NODEF
void HMM_GMM::CalBgOt(double **obs, int nframe, int dim)/*{{{*/
{
  vector<Gaussian *> &vGauss = *(getpGM(0,USE)->getpGaussPool());
  bgOt.resize(vGauss.size());
  for (unsigned g = 0; g < vGauss.size(); g++) {
    bgOt[g].resize(nframe);
    for (unsigned t = 0; t < nframe; t++)
      bgOt[g][t] = vGauss[g]->logProb(obs[t],dim,false);
  }
}/*}}}*/
void HMM_GMM::CalBjOtPxs(int nframe)/*{{{*/
{
  GaussianMixture *state;
  int mixsize;
  bjOt.resize(i_nstate);
  px_s.resize(i_nstate);
  /* For each state */
  for (int j = 0; j < i_nstate; j++) {
    state  = getpGM(j,USE);
    mixsize = state->getNmix();
    bjOt[j].resize(nframe);
    px_s[j].resize(mixsize);
    /* For each Gaussian */
    for (int x = 0; x < mixsize; x++) {
      px_s[j][x].resize(nframe);
      for (int t = 0; t < nframe; t++)
        px_s[j][x][t] = state->getWeight(x) * bgOt[state->getGaussIdx(x)][t];
      //px_s[j][x][t] = logprod(log(state->getWeight(x)), bgOt[state->getGaussIdx(x)][t]);
    }
    /* For each frame */
    for (int t = 0; t < nframe; t++) {
      bjOt[j][t] = 0.0;
      for (int x = 0; x < mixsize; x++)
        bjOt[j][t] +=  px_s[j][x][t];
      //bjOt[j][t] = logadd(bjOt[j][t], px_s[j][x][t]);
      for (int x = 0; x < mixsize; x++)
        px_s[j][x][t] /= bjOt[j][t];
      //px_s[j][x][t] = logdiv(px_s[j][x][t], bjOt[j][t]);
    }
  }
}/*}}}*/
void HMM_GMM::CalAlpha(int nframe)/*{{{*/
{

  alpha.resize(i_nstate);
  for (int i = 0; i < i_nstate; i++)
  {
    alpha[i].resize(nframe);
    alpha[i][0] = pi[use][i] * bjOt[i][0];
  }
  for (int t = 1; t < nframe; t++) {
    for (int j = 0; j < i_nstate; j++) {
      alpha[j][t] = 0.0;
      for (int i = 0; i < i_nstate; i++) {
        if (v_trans[i][j] < ZERO) continue;
        alpha[j][t] += alpha[i][t-1] * v_trans[i][j];
      }
      alpha[j][t] *= bjOt[j][t];
    }
  }
}/*}}}*/
void HMM_GMM::CalBeta(int nframe, vector<int> *p_state_seq)/*{{{*/
{
  beta.resize(i_nstate);
  for (int i = 0; i < i_nstate; i++) {
    beta[i].resize(nframe);
    if (getRight(i) == -1) beta[i][nframe-1] = 1.0;
    else beta[i][nframe-1] = 0.0;
  }
  vector<double> nextbeta;
  nextbeta.resize(i_nstate);
  for (int t = nframe-2; t >= 0; t--) {
    for (int j = 0; j < i_nstate; j++)
      nextbeta[j] = beta[j][t+1] * bjOt[j][t+1];
    for (int i = 0; i < i_nstate; i++) {
      beta[i][t] = 0.0;
      for (int j = 0; j < i_nstate; j++) {
        if (v_trans[i][j] < ZERO) continue;
        beta[i][t] += v_trans[i][j] * nextbeta[j];
      }
    }
  }
}/*}}}*/
void HMM_GMM::CalPrO(int nframe)/*{{{*/
{
  int nframe_1 = nframe - 1;
  prO = 0.0;
  for (int i = 0; i < i_nstate; i++)
    prO += alpha[i][nframe_1];

}/*}}}*/
void HMM_GMM::CalGamma(int nframe)/*{{{*/
{
  gamma.resize(i_nstate);
  for (int i = 0; i < i_nstate; i++) {
    gamma[i].resize(nframe);
    for (int t = 0; t < nframe; t++) {
      gamma[i][t] = alpha[i][t] * beta[i][t] / prO;
    }
  }

}/*}}}*/
void HMM_GMM::CalSumGamma()/*{{{*/
{
  sum_gamma.resize(i_nstate);
  for (int i = 0; i < i_nstate; i++) {
    sum_gamma[i] = 0.0;
    for (unsigned t = 0; t < gamma[i].size(); t++)
      sum_gamma[i] += gamma[i][t];
  }

}/*}}}*/
void HMM_GMM::CalEpsilon(int nframe)/*{{{*/
{
  int nframe_1 = nframe-1;
  epsilon.resize(i_nstate);
  for (int i = 0; i < i_nstate; i++) {
    epsilon[i].resize(i_nstate);
    for (int j = 0; j < i_nstate; j++)
      epsilon[i][j].resize(nframe_1);
  }

  for (int t = 0; t < nframe_1; t++) {
    for (int j = 0; j < i_nstate; j++) {
      double bjot_beta_prO = bjOt[j][t+1] * beta[j][t+1] / prO;
      for (int i = 0; i < i_nstate; i++) {
        if (v_trans[i][j] < ZERO) continue;
        epsilon[i][j][t] = alpha[i][t] * v_trans[i][j] * bjot_beta_prO;
      }
    }
  }

}/*}}}*/
#endif

void HMM_GMM::SyncUsed()/*{{{*/
{
  vector<Gaussian *> &vGauss = *getpGaussPool(USE);
  vector<GaussianMixture *> &vState = *getpStatePool(USE);

  state_isUsed.resize(vState.size());
  gauss_isUsed.resize(vGauss.size());

  for (unsigned g = 0; g < gauss_isUsed.size(); g++) gauss_isUsed[g] = false;
  for (unsigned i = 0; i < state_isUsed.size(); i++) state_isUsed[i] = false;

  for (int j = 0; j < i_nstate; j++) {
    state_isUsed[getGMidx(j)] = true;
    GaussianMixture *state  = getpGM(j,USE);
    int mixsize = state->getNmix();
    for (int x = 0; x < mixsize; x++)
      gauss_isUsed[state->getGaussIdx(x)] = true;
  }
}/*}}}*/

void HMM_GMM::CalLogBjOtPxs(int nframe)/*{{{*/
{
  GaussianMixture *state;
  int mixsize;
  bjOt.resize(i_nstate);
  px_s.resize(i_nstate);
  /* For each state */
  for (int j = 0; j < i_nstate; j++) {
    state  = getpGM(j,USE);
    mixsize = state->getNmix();
    bjOt[j].resize(nframe);
    px_s[j].resize(mixsize);
    /* For each Gaussian */
    for (int x = 0; x < mixsize; x++) {
      px_s[j][x].resize(nframe);
      /* For each frame */
      for (int t = 0; t < nframe; t++) {
        px_s[j][x][t] = LProd(state->getWeight(x),
                              bgOt[state->getGaussIdx(x)][t]);
      }
    }
    /* For each frame */
    for (int t = 0; t < nframe; t++) {
      bjOt[j][t] = LZERO;
      for (int x = 0; x < mixsize; x++)
        bjOt[j][t] = LAdd(bjOt[j][t], px_s[j][x][t]);
      for (int x = 0; x < mixsize; x++) {
        if (bjOt[j][t] > LSMALL) px_s[j][x][t] = LDiv(px_s[j][x][t], bjOt[j][t]);
        else px_s[j][x][t] = LZERO;
      }
    }
  }
}/*}}}*/


void HMM_GMM::CalLogCondToLogPostBjOt() {
  int nframe = static_cast<int>(bjOt[0].size());
  for (int t = 0; t < nframe; ++t) {
    double tot_prob = 0.0;
    for (int j = 0; j < i_nstate; ++j) {
      tot_prob += EXP(bjOt[j][t]);
    }
    tot_prob = LOG(tot_prob);
    for (int j = 0; j < i_nstate; ++j) {
      bjOt[j][t] -= tot_prob;
    }
  }
}

void HMM_GMM::CalLogCondToPostBjOt() {
  int nframe = static_cast<int>(bjOt[0].size());
  for (int t = 0; t < nframe; ++t) {
    double tot_prob = 0.0;
    for (int j = 0; j < i_nstate; ++j) {
      bjOt[j][t] = EXP(bjOt[j][t]);
      tot_prob += bjOt[j][t];
    }
    for (int j = 0; j < i_nstate; ++j) {
      bjOt[j][t] /= tot_prob;
    }
  }
}

void HMM_GMM::CalLogAlpha(int nframe, vector<int> *p_label)/*{{{*/
{
  bool useLabel = (p_label != NULL);

  /* Init size */
  if (useLabel) alpha.resize(p_label->size());
  else alpha.resize(i_nstate);

  /* Init size and alpha at t = 0 */
  for (unsigned i = 0; i < alpha.size(); i++)
  {
    alpha[i].resize(nframe);
    if (useLabel && static_cast<int>(i) > allowedNDel)
      alpha[i][0] = LZERO;
    else{
      int sno = useLabel ? (*p_label)[i] : i;
      alpha[i][0] = LProd(pi[use][sno],  bjOt[sno][0]);
    }
  }

  /* Fill the table */
  for (int t = 1; t < nframe; t++) {
    for (unsigned j = 0; j < alpha.size(); j++) {
      alpha[j][t] = LZERO;

      int i_start = useLabel ? max(0, static_cast<int>(j-allowedNDel-1)) : 0;
      int i_end   = useLabel ? j+1 : i_nstate;
      int sno_j   = useLabel ? (*p_label)[j] : j;

      for (int i = i_start; i < i_end; i++) {
        int sno_i = useLabel ? (*p_label)[i] : i;
        if (v_trans[sno_i][sno_j] <= LSMALL) continue;
        alpha[j][t] = LAdd(alpha[j][t],
                           LProd(alpha[i][t-1], v_trans[sno_i][sno_j]));
      }

      alpha[j][t] = LProd(alpha[j][t], bjOt[sno_j][t]);
    }
  }
}/*}}}*/

void HMM_GMM::CalLogAlphaBound(int nframe, vector<int> *p_endf)/*{{{*/
{

  vector<int> &endf = *p_endf;
  /* Init size */
  alpha.resize(i_nstate);

  /* Init size and alpha at t = 0 */
  for (unsigned i = 0; i < alpha.size(); i++)
  {
    alpha[i].resize(nframe);
    alpha[i][0] = LProd(pi[use][i],  bjOt[i][0]);
  }

  /* Fill the table */
  int i_end = 0;
  bool isBound;
  for (int t = 1; t < nframe; t++) {
    isBound = (t-1 == endf[i_end]);
    if (isBound) {
      //cout << "Touch bound " << t-1 << " - " << t << endl;
      i_end++;
    }
    for (unsigned j = 0; j < alpha.size(); j++) {
      alpha[j][t] = LZERO;
      // inside a segment
      if (!isBound) {
        alpha[j][t] = LProd(alpha[j][t-1],v_trans[j][j]);
        if (getLeft(j) != -1) { // Not a head
          for (int i = 0; i < i_nstate; i++) { // Search for all possible links
            if (i == static_cast<int>(j) || getRight(i) == -1 || v_trans[i][j] <= LSMALL) continue;
            alpha[j][t] = LAdd(alpha[j][t], LProd(alpha[i][t-1], v_trans[i][j]));
          }
        }
      }
      // between segments
      else{
        for (int i = 0; i < i_nstate; i++) {
          if (v_trans[i][j] <= LSMALL) continue;
          alpha[j][t] = LAdd(alpha[j][t], LProd(alpha[i][t-1], v_trans[i][j]));
        }
      }
      alpha[j][t] = LProd(alpha[j][t], bjOt[j][t]);
    }
  }
}/*}}}*/

void HMM_GMM::CalLogBeta(int nframe, vector<int> *p_label)/*{{{*/
{
  bool useLabel = (p_label != NULL);

  /* Init size */
  if (useLabel) beta.resize(p_label->size());
  else beta.resize(i_nstate);

  /* Init size and beta at end */
  for (unsigned i = 0; i < beta.size(); i++) {
    beta[i].resize(nframe);
    int sno = useLabel ? (*p_label)[i] : i;
    if (getRight(sno) != -1)
      beta[i][nframe-1] = LZERO;
    else if (useLabel && i < beta.size() - allowedNDel)
      beta[i][nframe-1] = LZERO;
    else beta[i][nframe-1] = 0.0;
  }

  /* nextbeta is used to avoid repeated calculation */
  vector<double> nextbeta;
  nextbeta.resize(beta.size());

  /* Fill the table */
  for (int t = nframe-2; t >= 0; t--) {
    /* Calculate nextbeta */
    for (unsigned j = 0; j < beta.size(); j++) {
      int sno_j = useLabel ? (*p_label)[j] : j;
      nextbeta[j] = LProd(beta[j][t+1], bjOt[sno_j][t+1]);
    }

    for (unsigned i = 0; i < beta.size(); i++) {
      beta[i][t] = LZERO;

      int j_start = useLabel ? i : 0;
      int j_end   = useLabel ? min(static_cast<unsigned>(beta.size()), i+allowedNDel+1) : beta.size();
      int sno_i   = useLabel ? (*p_label)[i] : i;

      for (int j = j_start; j < j_end; j++) {
        int sno_j = useLabel ? (*p_label)[j] : j;
        if (v_trans[sno_i][sno_j] <= LSMALL) continue;
        beta[i][t] = LAdd(beta[i][t], LProd(v_trans[sno_i][sno_j], nextbeta[j]));
      }
    }
  }
}/*}}}*/

void HMM_GMM::CalLogBetaBound(int nframe, vector<int> *p_startf)/*{{{*/
{

  vector<int> &startf = *p_startf;
  /* Init size */
  beta.resize(i_nstate);

  /* Init size and beta at end */
  for (unsigned i = 0; i < beta.size(); i++) {
    beta[i].resize(nframe);
    if (getRight(i) != -1) beta[i][nframe-1] = LZERO;
    else beta[i][nframe-1] = 0.0;
  }

  /* nextbeta is used to avoid repeated calculation */
  vector<double> nextbeta;
  nextbeta.resize(beta.size());

  int i_start = startf.size()-1;
  /* Fill the table */
  for (int t = nframe-2; t >= 0; t--) {
    bool isBound = (t+1 == startf[i_start]);
    if (isBound) {
      //cout << "Touch bound " << t << " - " << t+1 << endl;
      i_start--;
    }
    /* Calculate nextbeta */
    for (unsigned j = 0; j < beta.size(); j++) {
      nextbeta[j] = LProd(beta[j][t+1], bjOt[j][t+1]);
    }

    for (unsigned i = 0; i < beta.size(); i++) {
      beta[i][t] = LZERO;
      // inside a segment
      if (!isBound) {
        beta[i][t] = LProd(v_trans[i][i], nextbeta[i]);
        if (getRight(i) != -1) { // Not a tail
          for (int j = 0; j < i_nstate; j++) {
            if (static_cast<int>(i) == j || getLeft(j) == -1 || v_trans[i][j] <= LSMALL) continue;
            beta[i][t] = LAdd(beta[i][t], LProd(v_trans[i][j], nextbeta[j]));
          }
        }
      }
      // between segments
      else{
        for (int j = 0; j < i_nstate; j++) {
          if (v_trans[i][j] <= LSMALL) continue;
          beta[i][t] = LAdd(beta[i][t],
                            LProd(v_trans[i][j], nextbeta[j]));
        }
      }
    }

  }
}/*}}}*/

void HMM_GMM::CalLogPrO(const int nframe, vector<int> *p_label)/*{{{*/
{
  bool useLabel = (p_label != NULL);
  int nframe_1 = nframe - 1;
  prO = LZERO;
  /*
     for (unsigned i = 0; i < alpha.size(); i++)
     prO = LAdd(prO, LProd(alpha[i][nframe_1], beta[i][nframe_1]));
     */
  for (unsigned i = 0; i < alpha.size(); i++) {
    int sno = (useLabel) ? (*p_label)[i] : i;
    if (getRight(sno) != -1) continue;
    if (useLabel && i < alpha.size() - allowedNDel) continue;
    prO = LAdd(prO, alpha[i][nframe_1]);
  }
  if (prO < LSMALL) {
    fprintf(stderr,"Warning: LogProb = zero\n");
    printVVec(bgOt,"bgOt");
    printVVec(alpha,"alpha");
    printVVec(gamma,"gamma");
  }

}/*}}}*/

void HMM_GMM::CalLogGamma(int nframe)/*{{{*/
{
  gamma.resize(alpha.size());
  for (unsigned i = 0; i < gamma.size(); i++) {
    gamma[i].resize(nframe);
    for (int t = 0; t < nframe; t++) {
      gamma[i][t] = LDiv(LProd(alpha[i][t], beta[i][t]), prO);
    }
  }

}/*}}}*/

void HMM_GMM::CalLogEpsilon(int nframe, vector<int> *p_label)/*{{{*/
{
  bool useLabel = (p_label != NULL);
  int nframe_1 = nframe-1;
  epsilon.resize(alpha.size());
  for (unsigned i = 0; i < epsilon.size(); i++) {
    epsilon[i].resize(alpha.size());
    for (unsigned j = 0; j < epsilon[i].size(); j++)
      epsilon[i][j].resize(nframe_1);
  }

  for (int t = 0; t < nframe_1; t++) {
    for (unsigned j = 0; j < epsilon.size(); j++) {
      int sno_j = useLabel ? (*p_label)[j] : j;
      double bjot_beta_prO = LDiv(LProd(bjOt[sno_j][t+1], beta[j][t+1]), prO);
      int i_start = useLabel ? max(0u,j-allowedNDel-1) : 0;
      int i_end   = useLabel ? j+1 : i_nstate;
      for (int i = i_start; i < i_end; i++) {
        int sno_i = useLabel ? (*p_label)[i] : i;
        if (v_trans[sno_i][sno_j] < LSMALL) {
          epsilon[i][j][t] = LZERO;
          continue;
        }
        epsilon[i][j][t] = LProd(LProd(alpha[i][t], v_trans[sno_i][sno_j]), bjot_beta_prO);
      }
    }
  }

}/*}}}*/

void HMM_GMM::CalLogEpsilonBound(int nframe, vector<int> *p_endf)/*{{{*/
{
  vector<int> &endf = *p_endf;
  /* Allocate memory */
  int nframe_1 = nframe-1;
  epsilon.resize(i_nstate);
  for (unsigned i = 0; i < epsilon.size(); i++) {
    epsilon[i].resize(i_nstate);
    for (unsigned j = 0; j < epsilon[i].size(); j++)
      epsilon[i][j].resize(nframe_1);
  }

  /*                               / Y: LZERO        (case2)
   * isBound / Y: v_trans < LSMALL \ N: OK           (case3)
   *         \ N: j == i || j == Right(i) / Y: OK    (case3)
   *                                      \ N: LZERO (case1)
   */

  /* Fill table */
  int i_end = 0;
  for (int t = 0; t < nframe_1; t++) {
    bool isBound = (t == endf[i_end]);
    if (isBound) i_end++;
    for (int j = 0; j < i_nstate; j++) {
      double bjot_beta_prO = LDiv(LProd(bjOt[j][t+1], beta[j][t+1]), prO);
      for (int i = 0; i < i_nstate; i++) {
        if (isBound || i == j || (getRight(i) != -1 && getLeft(j) != -1)) {
          if (v_trans[i][j] > LSMALL)
            epsilon[i][j][t] = LProd(LProd(alpha[i][t], v_trans[i][j]), bjot_beta_prO);
          else
            epsilon[i][j][t] = LZERO;
        }
        else epsilon[i][j][t] = LZERO;
      }
    }
  }

}/*}}}*/

void HMM_GMM::AccumPi(vector<int> *p_label, double obs_weight)/*{{{*/
{
  int unuse = 1-use;
  bool useLabel = (p_label != NULL);
  int sno;
  pthread_mutex_lock(&H_mutex);
  for (unsigned i = 0; i < gamma.size(); i++) {
    if (gamma[i][0] < ZERO) continue;
    sno = useLabel ? (*p_label)[i] : i;
    pi[unuse][sno] += obs_weight * gamma[i][0];
  }
  pthread_mutex_unlock(&H_mutex);
}/*}}}*/

void HMM_GMM::AccumIJ(int nframe, vector<int> *p_label, double obs_weight)/*{{{*/
{
  int nframe_1 = nframe - 1;
  bool useLabel = (p_label != NULL);

  pthread_mutex_lock(&H_mutex);

  for (unsigned i = 0; i < epsilon.size(); i++) {
    int sno_i = useLabel ? (*p_label)[i] : i;
    int j_start = useLabel ? i : 0;
    int j_end   = useLabel ? min(static_cast<unsigned>(epsilon[i].size()), i+allowedNDel+1) : epsilon[i].size();
    for (int j = j_start; j < j_end; j++) {
      int sno_j = useLabel ? (*p_label)[j] : j;
      if (v_trans[sno_i][sno_j] < LSMALL) continue;
      for (int t = 0; t < nframe_1; t++)
        // We do not accumulate gamma[i][:] because
        // the sum of accum_ij[i][:] = gamma[i][:].
        // Hence normalize accum_ij is actually dividing
        // the sum of gamma[i][:]
        accum_ij[sno_i][sno_j] += obs_weight * epsilon[i][j][t];
    }
  }
  pthread_mutex_unlock(&H_mutex);
}/*}}}*/

void HMM_GMM::AccumWeightGaussian(double **obs, int nframe, int dim, UpdateType udtype, vector<int> *p_label, double obs_weight)/*{{{*/
{
  bool useLabel = (p_label != NULL);

  /* For each state */
  for (unsigned i = 0; i < gamma.size(); i++) {
    int sno_i = useLabel ? (*p_label)[i] : i;
    GaussianMixture *state = getpGM(sno_i,UNUSE);
    /* For each gaussian associated with this state*/
    for (int x = 0; x < state->getNmix(); x++) {
      Gaussian *pg = state->getpGauss(x);
      for (int t = 0; t < nframe; t++) {
        double gamma_i_x_t = obs_weight * gamma[i][t] * px_s[sno_i][x][t];
        if (gamma_i_x_t <= ZERO) continue;
        pg->AddData(obs[t],dim,gamma_i_x_t,udtype);
        // Here we do not divide by sum_gamma[i] because
        // the total sum of weights in state i is
        // actually sum_gamma[i]
        state->setWeight(x,gamma_i_x_t, ADD);
      }
    }
  }

}/*}}}*/

double HMM_GMM::normPi(UseType u)/*{{{*/
{
  double sum = 0.0;
  int w = (u == USE) ? use : 1-use;
  for (int i = 0; i < i_nstate; i++)
    sum += pi[w][i];
  for (int i = 0; i < i_nstate; i++)
    pi[w][i] /= sum;
  return sum;

}/*}}}*/

void HMM_GMM::normTrans()/*{{{*/
{
  for (int i = 0; i < i_nstate; i++) {
    double sum = 0.0;
    for (int j = 0; j < i_nstate; j++) {
      if (v_trans[i][j] < ZERO) v_trans[i][j] = 0.0;
      else sum += v_trans[i][j];
    }
    double trans_floor = (sum-v_trans[i][i]) * TRANS_ZERO / (i_nstate-1);
    sum = 0.0;
    for (int j = 0; j < i_nstate; j++) {
      if (v_trans[i][j] < ZERO) continue;
      if (v_trans[i][j] < trans_floor) {
        v_trans[i][j] = 0.0;
      }
      else sum += v_trans[i][j];
    }
    if (sum < ZERO) {
      cerr << "no occupation for state no " << i << ", transition set to zero" << endl;
      for (int j = 0; j < i_nstate; j++) v_trans[i][j] = 0.0;
    }
    for (int j = 0; j < i_nstate; j++) {
      if (v_trans[i][j] > ZERO) v_trans[i][j] /= sum;
    }
  }
}/*}}}*/

void HMM_GMM::normTransOther()/*{{{*/
{
  for (int i = 0; i < i_nstate; i++) {
    double sum = 0.0;
    for (int j = 0; j < i_nstate; j++) {
      if (i == j) continue;
      else if (v_trans[i][j] < ZERO) v_trans[i][j] = 0.0;
      else sum += v_trans[i][j];
    }
    if (sum < ZERO)
      v_trans[i][i] = 1.0;
    else {
      double ratio = (1 - v_trans[i][i]) / sum;
      for (int j = 0; j < i_nstate; j++) {
        if (i == j || v_trans[i][j] < ZERO) continue;
        v_trans[i][j] *= ratio;
      }
    }
  }
}/*}}}*/

void HMM_GMM::normRTrans()/*{{{*/
{
  for (int j = 0; j < i_nstate; j++) {
    double sum = 0.0;
    for (int i = 0; i < i_nstate; i++) {
      if (v_rtrans[i][j] < ZERO) v_rtrans[i][j] = 0.0;
      else sum += v_rtrans[i][j];
    }
    double trans_floor = sum * TRANS_ZERO / i_nstate;
    for (int i = 0; i < i_nstate; i++) {
      if (v_rtrans[i][j] < ZERO) continue;
      if (v_rtrans[i][j] < trans_floor) {
        sum -= v_rtrans[i][j];
        v_rtrans[i][j] = 0.0;
      }
    }
    for (int i = 0; i < i_nstate; i++) {
      if (v_rtrans[i][j] > ZERO) v_rtrans[i][j] /= sum;
    }
  }
}/*}}}*/

void HMM_GMM::accum2Trans()/*{{{*/
{
  for (int i = 0; i < i_nstate; i++)
    for (int j = 0; j < i_nstate; j++)
      if (accum_ij[i][j] < ZERO) accum_ij[i][j] = 0.0;

  for (int i = 0; i < i_nstate; i++) {
    double sum = 0.0;
    for (int j = 0; j < i_nstate; j++) {
      if (accum_ij[i][j] > ZERO) sum += accum_ij[i][j];
    }
    double trans_floor = (sum - accum_ij[i][i]) * TRANS_ZERO / (i_nstate-1);
    sum = 0.0;
    for (int j = 0; j < i_nstate; j++) {
      if (accum_ij[i][j] >= trans_floor) sum += accum_ij[i][j];
    }
    if (sum < ZERO) {
      cerr << "no occupation for state no " << i << ", transition set to zero" << endl;
      for (int j = 0; j < i_nstate; j++) v_trans[i][j] = 0.0;
    }
    else{
      for (int j = 0; j < i_nstate; j++) {
        if (accum_ij[i][j] >= trans_floor)
          v_trans[i][j] = accum_ij[i][j] / sum;
        else v_trans[i][j] = 0.0;
      }
    }
  }

  for (int j = 0; j < i_nstate; j++) {
    double sum = 0.0;
    for (int i = 0; i < i_nstate; i++) {
      if (accum_ij[i][j] > ZERO) sum += accum_ij[i][j];
    }
    double trans_floor = sum * TRANS_ZERO / i_nstate;
    sum = 0.0;
    for (int i = 0; i < i_nstate; i++) {
      if (accum_ij[i][j] >= trans_floor) sum += accum_ij[i][j];
    }
    for (int i = 0; i < i_nstate; i++) {
      if (accum_ij[i][j] >= trans_floor)
        v_rtrans[i][j] = accum_ij[i][j] / sum;
      else v_rtrans[i][j] = 0.0;
    }
  }

}/*}}}*/

double HMM_GMM::normOccupation()/*{{{*/
{
  double total_occ = 0.0;
  for (unsigned i = 0; i < occupation.size(); i++)
    total_occ += occupation[i];
  for (unsigned i = 0; i < occupation.size(); i++)
    occupation[i] /= total_occ;
  return total_occ;

}/*}}}*/

void HMM_GMM::dump_var()/*{{{*/
{
  cout << "==== EM variables ====\n";
  /* bgOt */
  cout << "bgOt" << endl;
  for (unsigned g = 0; g < bgOt.size(); g++) {
    for (unsigned t = 0; t < bgOt[g].size(); t++)
      cout << setprecision(3) << bgOt[g][t] << ' ';
    cout << endl;
  }
  /* bjOt */
  cout << "bjOt" << endl;
  for (unsigned i = 0; i < bjOt.size(); i++) {
    for (unsigned t = 0; t < bjOt[i].size(); t++)
      cout << setprecision(3) << bjOt[i][t] << ' ';
    cout << endl;
  }
  /* px_s */
  cout << "px_s" << endl;
  for (unsigned i = 0; i < px_s.size(); i++) {
    for (unsigned t = 0; t < bjOt[i].size(); t++) {
      cout << "{";
      for (unsigned x = 0; x < px_s[i].size(); x++)
        cout << setprecision(3) << px_s[i][x][t] << ' ';
      cout << "} ";
    }
    cout << endl;
  }
  /* alpha */
  cout << "alpha" << endl;
  for (unsigned i = 0; i < alpha.size(); i++) {
    for (unsigned t = 0; t < alpha[i].size(); t++)
      cout << setprecision(3) << alpha[i][t] << ' ';
    cout << endl;
  }
  /* beta */
  cout << "beta" << endl;
  for (unsigned i = 0; i < beta.size(); i++) {
    for (unsigned t = 0; t < beta[i].size(); t++)
      cout << setprecision(3) << beta[i][t] << ' ';
    cout << endl;
  }
  /* gamma */
  cout << "gamma" << endl;
  for (unsigned i = 0; i < gamma.size(); i++) {
    for (unsigned t = 0; t < gamma[i].size(); t++)
      cout << setprecision(3) << gamma[i][t] << ' ';
    cout << endl;
  }

}/*}}}*/

void HMM_GMM::EMUpdate(set<int> *p_delete_list, double backoff_weight, UpdateType udtype)/*{{{*/
{
  assert(backoff_weight >= 0 && backoff_weight <= 1);
  /**** Pi ****/
  normPi(UNUSE);
  /**** Trans ****/
  accum2Trans();
  /**** Mixture weight ****/
  occupation.resize(i_nstate);
  for (int i = 0; i < i_nstate; i++) {
    GaussianMixture *state = getpGM(i,UNUSE);
    if (!state->normWeight(occupation[i])) {
      GaussianMixture *oldstate = getpGM(i,USE);
      state->copyWeight(*oldstate);
      if (p_delete_list != NULL)
        p_delete_list->insert(getGMidx(i));
    }
  }
  /**** Gaussians ****/
  vector<Gaussian *> &vGauss = *(getpGM(0,UNUSE)->getpGaussPool());
  vector<Gaussian *> &vGauss_old = *(getpGM(0,USE)->getpGaussPool());
  //bool enoughTrain;
  for (unsigned g = 0; g < vGauss.size(); g++) {
    if (!gauss_isUsed[g]) continue;
    // all backoff
    if (backoff_weight > 1 - ZERO) {
      cerr << "Gaussian[" << g << "] not updated\n";
      *(vGauss[g]) = *(vGauss_old[g]);
      continue;
    }
    // 1. Take care of unupdated part
    if (udtype == UpdateCov)
      vGauss[g]->CopyMean(*vGauss_old[g]);
    else if (udtype == UpdateCov)
      vGauss[g]->CopyCov(*vGauss_old[g]);

    // 2. The rest is taken cared of by normMeanCov()
    if (!vGauss[g]->normMeanCov(true, udtype)) {
      // Not enough training data
      cerr << "Gaussian[" << g << "] forced backoff.\n";
      backoff_weight = max(0.7, backoff_weight);
    }

    vGauss[g]->AddVarFloor();
    //double totvar = vGauss[g]->getTotalVar();
    //double varfloor = Gaussian::getVarFloor() * vGauss[g]->getDim();
    //fprintf(stdout, "vfloor contribution = %g/%g(%g)\n", varfloor, totvar, varfloor/totvar);
    if (backoff_weight > ZERO)
      vGauss[g]->backoff(*vGauss_old[g], backoff_weight);
    if (vGauss[g]->InvertCov() == std::numeric_limits<double>::infinity()) {
      cerr << "Gaussian[" << g << "] singular\n";
      *(vGauss[g]) = *(vGauss_old[g]);
    }
  }
  use = 1 - use;
  isLog = false;

}/*}}}*/

double HMM_GMM::EMObs(double **obs, int nframe, int dim, double obs_weight, UpdateType udtype) /*{{{*/
{
  CalLogBgOt(obs,nframe,dim); // Use gauss_isUsed; +bgOt
  CalLogBjOtPxs(nframe);      // Use bgOt; +bjOt, +px_s
  CalLogAlpha(nframe);        // Use trans pi bjot; +alpha
  CalLogBeta(nframe);         // Use trans bjot; +beta
  CalLogPrO(nframe);          // Use alpha; +prO;
  CalLogGamma(nframe);        // Use alpha beta prO; +gamma
  CalLogEpsilon(nframe);      // Use alpha beta trans bjot prO; +epsilon
  ExpPxs();
  ExpGamma();
  ExpEpsilon();
  AccumPi(NULL,obs_weight);        // Use gamma; +pi
  AccumIJ(nframe,NULL,obs_weight); // Use epsilon; +acuum_ij
  AccumWeightGaussian(obs,nframe,dim,udtype,NULL,obs_weight); // Use gamma px_s; (+e)state->setWeight(), (+e)gauss->AddData()

  return obs_weight * prO;
}/*}}}*/

double HMM_GMM::EMObsLabel(double **obs, int nframe, int dim, vector<int> *p_label, double obs_weight, UpdateType udtype) /*{{{*/
{
  Labfile labfile;
  bool newLabel = (p_label == NULL);
  if (newLabel) {
    p_label = new vector<int>;
  }

  CalLogBgOt(obs,nframe,dim);            // Use gauss_isUsed; +bgOt
  CalLogBjOtPxs(nframe);                 // Use bgOt; +bjOt, +px_s

  if (newLabel) {
    CalLogDelta(*p_label);             // Use bjOt; +delta
    labfile.parseStateSeq(*p_label);
    delete p_label;
    p_label = labfile.getpCluster();
  }
  for (unsigned i = 0; i < p_label->size()-1; i++) {
    bool can_trans = false;
    int sno_i = (*p_label)[i];
    for (unsigned j = i+1; j-i-1 <= static_cast<unsigned>(allowedNDel) && j < p_label->size(); j++) {
      int sno_j = (*p_label)[j];
      if (v_trans[ sno_i ][ sno_j ] > LSMALL) {
        can_trans = true;
        break;
      }
    }
    if (!can_trans) {
      cerr << "\nWarning: transition from label[" << i << "](" << sno_i << ") is impossible!\n";
      for (unsigned j = i+1; j-i-1 <= static_cast<unsigned>(allowedNDel) && j < p_label->size(); j++) {
        int sno_j = (*p_label)[j];
        cerr << "trans(" << sno_i << ", " << sno_j << ") = " << v_trans[sno_i][sno_j] << " < LSMALL = " << LSMALL << endl;
      }
    }
  }
  CalLogAlpha(nframe,p_label);       // Use trans pi bjot; +alpha
  CalLogBeta(nframe,p_label);        // Use trans bjot; +beta
  CalLogPrO(nframe,p_label);         // Use alpha; +prO;
  CalLogGamma(nframe);               // Use alpha beta prO; +gamma
  CalLogEpsilon(nframe,p_label);     // Use alpha beta trans bjot prO; +epsilon
  ExpPxs();
  ExpGamma();
  ExpEpsilon();
  AccumPi(p_label,obs_weight);       // Use gamma; +pi
  AccumIJ(nframe,p_label,obs_weight);// Use epsilon; +acuum_ij
  AccumWeightGaussian(obs,nframe,dim,udtype,p_label,obs_weight); // Use gamma px_s; (+e)state->setWeight(), (+e)gauss->AddData()

  return obs_weight * prO;
}/*}}}*/

double HMM_GMM::EMObsBound(double **obs, int nframe, int dim, Labfile *p_reflabfile, double obs_weight, UpdateType udtype) /*{{{*/
{

  CalLogBgOt(obs,nframe,dim);    // Use gauss_isUsed; +bgOt
  CalLogBjOtPxs(nframe);         // Use bgOt; +bjOt, +px_s
  CalLogAlphaBound(nframe,p_reflabfile->getpEndf());   // Use trans pi bjot; +alpha
  CalLogBetaBound(nframe,p_reflabfile->getpStartf());  // Use trans bjot; +beta
  CalLogPrO(nframe);             // Use alpha; +prO;
  CalLogGamma(nframe);           // Use alpha beta prO; +gamma
  CalLogEpsilonBound(nframe,p_reflabfile->getpEndf()); // Use alpha beta trans bjot prO; +epsilon
  ExpPxs();
  ExpGamma();
  ExpEpsilon();
  AccumPi(NULL,obs_weight);           // Use gamma; +pi
  AccumIJ(nframe,NULL,obs_weight);    // Use epsilon; +acuum_ij
  AccumWeightGaussian(obs,nframe,dim,udtype,NULL,obs_weight); // Use gamma px_s; (+e)state->setWeight(), (+e)gauss->AddData()

  return obs_weight * prO;
}/*}}}*/

void HMM_GMM::AccumFromThread(const HMM_GMM &model)/*{{{*/
{
  pthread_mutex_lock(&H_mutex);

  int unuse = 1 - use;
  for (int i = 0; i < i_nstate; i++) pi[unuse][i] += model.pi[unuse][i];
  for (int i = 0; i < i_nstate; i++)
    for (int j = 0; j < i_nstate; j++)
      accum_ij[i][j] += model.accum_ij[i][j];

  pthread_mutex_unlock(&H_mutex);

}/*}}}*/

double HMM_GMM::CalDelta(vector<int> &state_seq, bool isEnd)/*{{{*/
{
  vector<vector<int> >path;
  int nframe = bjOt[0].size();
  int nframe_1 = nframe - 1;
  delta.resize(i_nstate);
  path.resize(i_nstate);
  for (int i = 0; i < i_nstate; i++)
  {
    delta[i].resize(nframe);
    path[i].resize(nframe);
    delta[i][0] = pi[use][i] * bjOt[i][0];
  }
  for (int t = 1; t < nframe; t++) {
    for (int j = 0; j < i_nstate; j++) {
      delta[j][t] = 0.0;
      path[j][t] = 0;
      for (int i = 0; i < i_nstate; i++) {
        if (v_trans[i][j] < ZERO) continue;
        double newscore = delta[i][t-1] * v_trans[i][j];
        if (newscore > delta[j][t]) {
          delta[j][t] = newscore;
          path[j][t] = i;
        }
      }
      delta[j][t] *= bjOt[j][t];
    }
  }
  /* backtrack */
  state_seq.resize(nframe);
  state_seq[nframe_1] = i_nstate-1;
  double BestScore = delta.back().back();
  if (!isEnd) {
    for (int i = i_nstate - 2; i >= 0; i--) {
      if (delta[i][nframe_1] > BestScore) {
        BestScore = delta[i][nframe_1];
        state_seq[nframe_1] = i;
      }
    }
  }
  for (int t = nframe - 2; t >= 0; t--) {
    state_seq[t] = path[state_seq[t+1]][t+1];
  }
  return BestScore;
}/*}}}*/

double HMM_GMM::CalLogDelta(vector<int> &state_seq, const vector<int> *p_endf) {/*{{{*/
  bool bndConstraint = (p_endf != NULL);
  vector<vector<int> > path;
  int nframe = bjOt[0].size();
  int nframe_1 = nframe - 1;
  /* Allocate memory */
  delta.resize(i_nstate);
  path.resize(i_nstate);
  for (int i = 0; i < i_nstate; i++) {
    delta[i].resize(nframe);
    path[i].resize(nframe);
    delta[i][0] = LProd(pi[use][i], bjOt[i][0]);
  }
  /* Fill the table */
  int i_end = 0;
  bool isBound = true;
  int left_i;
  double newscore;
  for (int t = 1; t < nframe; t++) {
    if (bndConstraint) {
      isBound = (t-1 == (*p_endf)[i_end]);
      if (isBound) i_end++;
    }
    for (int j = 0; j < i_nstate; j++) {
      delta[j][t] = LZERO;
      // inside a segment
      if (!isBound) {
        delta[j][t] = LProd(delta[j][t-1], v_trans[j][j]);
        path[j][t] = j;
        if ((left_i = getLeft(j)) != -1) {
          newscore = LProd(delta[left_i][t-1], v_trans[left_i][j]);
          if (newscore > delta[j][t]) {
            delta[j][t] = newscore;
            path[j][t] = left_i;
          }
        }
      }
      // between segments
      else {
        for (int i = 0; i < i_nstate; i++) {
          if (v_trans[i][j] <= LSMALL) continue;
          newscore = LProd(delta[i][t-1], v_trans[i][j]);
          if (newscore > delta[j][t]) {
            delta[j][t] = newscore;
            path[j][t] = i;
          }
        }
      }
      delta[j][t] = LProd(delta[j][t], bjOt[j][t]);
    }
  }
  /* backtrack */
  state_seq.resize(nframe);
  //state_seq[nframe_1] = i_nstate-1;
  double BestScore = LZERO;
  for (int i = i_nstate - 1; i >= 0; i--) {
    if (getRight(i) == -1 && delta[i][nframe_1] > BestScore) {
      BestScore = delta[i][nframe_1];
      state_seq[nframe_1] = i;
      //cout << "BestScore @ " << state_seq[nframe_1] << endl;
    }
  }
  for (int t = nframe - 2; t >= 0; t--) {
    state_seq[t] = path[state_seq[t+1]][t+1];
  }
  return BestScore;
}/*}}}*/

#ifdef __TRAJECTORY__
double HMM_GMM::CalLogDeltaTrajectory(vector<double> &state_seq) {/*{{{*/
  /* Init transition states (2 states per transition)
     tranState parameters: (a'S, a'Sb, a'Sa) */

}/*}}}*/
#endif /* __TRAJECTORY__ */

/****** I/O ******/

void Gaussian::SaveGaussian(FILE *fp, const DataType type)/*{{{*/
{
  if (type == ASCII) {
    display(fp);
    return;
  }
  fprintf(fp,"Gaussian binary\n");

  fprintf(fp,"dim: %d\n",dim);

  fprintf(fp,"mean: ");
  assert(fwrite(p_mean->pointer(),sizeof(double),dim,fp) == static_cast<unsigned>(dim));

  fprintf(fp,"cov:\n");
  double *ptr = p_cov->pointer();
  for (int i = 0; i < dim; i++) {
    assert(fwrite(ptr,sizeof(double),i+1,fp)==i+1u);
    ptr += dim;
  }
  fprintf(fp,"EndGaussian\n");
}/*}}}*/

void Gaussian::LoadGaussian(FILE *fp)/*{{{*/
{
  char buff[1024];
  fscanf(fp,"%s",buff);
  assert(strcmp(buff,"Gaussian") == 0);
  fscanf(fp,"%s",buff);
  if (strcmp(buff,"ascii")==0) {
    ReadAscii(fp);
  }else if (strcmp(buff,"binary")==0) {
    ReadBinary(fp);
  }else{
    fprintf(stderr,"Unknown tag for Gaussian, only ascii/binary allowed\n");
  }
}/*}}}*/

void Gaussian::ReadAscii(FILE *fp)/*{{{*/
{
  char tag[1024];
  int i_val;
  float f_val;

  while (gettag(fp, tag)) {
    if (strcmp(tag,"EndGaussian") == 0) {
      break;
    }
    else if (strcmp(tag,"dim:") == 0) {
      fscanf(fp,"%d",&i_val);
      AllocateMem(i_val);
    }
    else if (strcmp(tag,"mean:") == 0) {
      for (int r = 0; r < dim; r++) {
        fscanf(fp,"%f",&f_val);
        p_mean->setEntry(r,0,static_cast<double>(f_val));
      }
    }
    else if (strcmp(tag,"cov:") == 0) {
      for (int r = 0; r < dim; r++) {
        for (int c = r; c < dim; c++) {
          fscanf(fp,"%f",&f_val);
          p_cov->setEntry(r,c,static_cast<double>(f_val));
        }
      }
    }
    else fprintf(stderr,"unknown tag ``%s''\n",tag);
  }
}/*}}}*/

void Gaussian::ReadBinary(FILE *fp)/*{{{*/
{
  char tag[1024];
  int i_val;

  while (gettag(fp, tag)) {
    if (strcmp(tag,"EndGaussian") == 0) {
      break;
    }
    else if (strcmp(tag,"dim:") == 0) {
      fscanf(fp,"%d",&i_val);
      AllocateMem(i_val);
    }
    else if (strcmp(tag,"mean:") == 0) {
      fgetc(fp);
      assert(fread(p_mean->pointer(),sizeof(double),dim,fp)==static_cast<unsigned>(dim));
    }
    else if (strcmp(tag,"cov:") == 0) {
      fgetc(fp);
      double *ptr = p_cov->pointer();
      for (int i = 0; i < dim; i++) {
        assert(fread(ptr,sizeof(double),i+1,fp)==i+1u);
        ptr += dim;
      }
    }
    else fprintf(stderr,"unknown tag ``%s''\n",tag);
  }
}/*}}}*/

void GaussianMixture::SaveGaussianMixture(FILE *fp, const DataType type)/*{{{*/
{
  if (type == BINARY)
    fprintf(stderr,"GaussianMixture::SaveGaussianMixture(): does not support binary\n");
  display(fp);
}/*}}}*/

void GaussianMixture::LoadGaussianMixture(FILE *fp)/*{{{*/
{
  char buff[1024];
  fscanf(fp,"%s",buff);
  assert(strcmp(buff,"GaussianMixture") == 0);
  fscanf(fp,"%s",buff);
  if (strcmp(buff,"ascii")==0) {
    ReadAscii(fp);
  }else if (strcmp(buff,"binary")==0) {
    ReadBinary(fp);
  }else{
    fprintf(stderr,"Unknown tag for GaussianMixture, only ascii/binary allowed\n");
  }
}/*}}}*/

void GaussianMixture::ReadAscii(FILE *fp)/*{{{*/
{
  char tag[1024];
  int i_val;
  float f_val;

  while (gettag(fp, tag)) {
    if (strcmp(tag,"EndGaussianMixture") == 0) {
      break;
    }
    else if (strcmp(tag,"dim:") == 0) {
      fscanf(fp,"%d",&dim);

    }
    else if (strcmp(tag,"nmix:") == 0) {
      fscanf(fp,"%d",&i_val);
      setNmix(i_val);
    }
    else if (strcmp(tag,"weight:") == 0) {
      for (unsigned x = 0; x < v_weight.size(); x++) {
        fscanf(fp,"%g",&f_val);
        v_weight[x] = static_cast<double>(f_val);
      }
    }
    else if (strcmp(tag,"gaussidx:") == 0) {
      for (unsigned x = 0; x < v_gaussidx.size(); x++)
        fscanf(fp,"%d",&v_gaussidx[x]);
    }
    else fprintf(stderr,"unknown tag ``%s''\n",tag);
  }
}/*}}}*/

void GaussianMixture::ReadBinary(FILE *fp)/*{{{*/
{
  fprintf(stderr,"Error: GaussianMixture::ReadBinary(FILE *fp) is not implemented.\n");
}/*}}}*/

void HMM_GMM::SaveHMM(FILE *fp, const DataType type)/*{{{*/
{
  if (type == BINARY)
    fprintf(stderr,"HMM_GMM::SaveHMM(): does not support binary\n");
  display(fp);
}/*}}}*/

void HMM_GMM::LoadHMM(FILE *fp)/*{{{*/
{
  char buff[1024];
  fscanf(fp,"%s",buff);
  assert(strcmp(buff,"HMM") == 0);
  fscanf(fp,"%s",buff);
  if (strcmp(buff,"ascii")==0) {
    ReadAscii(fp);
  }else if (strcmp(buff,"binary")==0) {
    ReadBinary(fp);
  }else{
    fprintf(stderr,"Unknown tag for HMM, only ascii/binary allowed\n");
  }
}/*}}}*/

void HMM_GMM::ReadBinary(FILE *fp)/*{{{*/
{
  fprintf(stderr,"Error: HMM_GMM::ReadBinary(FILE *fp) is not implemented.\n");
}/*}}}*/

void HMM_GMM::ReadAscii(FILE *fp)/*{{{*/
{
  char tag[1024];
  int i_val;
  float f_val;

  while (gettag(fp, tag)) {
    if (strcmp(tag,"EndHMM") == 0) {
      break;
    }
    else if (strcmp(tag,"pdf_weight:") == 0) {
      fscanf(fp,"%f",&f_val);
      set_pdf_weight(f_val);
    }
    else if (strcmp(tag,"nstate:") == 0) {
      fscanf(fp,"%d",&i_val);
      setNstate(i_val);
    }
    else if (strcmp(tag,"state:") == 0) {
      for (unsigned i = 0; i < v_state.size(); i++)
        fscanf(fp,"%d",&v_state[i]);
    }
    else if (strcmp(tag,"left:") == 0) {
      for (unsigned i = 0; i < left.size(); i++)
        fscanf(fp,"%d",&left[i]);
    }
    else if (strcmp(tag,"right:") == 0) {
      for (unsigned i = 0; i < left.size(); i++)
        fscanf(fp,"%d",&right[i]);
    }
    else if (strcmp(tag,"pi:") == 0) {
      for (unsigned i = 0; i < pi[use].size(); i++) {
        fscanf(fp,"%f",&f_val);
        pi[use][i] = f_val;
      }
    }
    else if (strcmp(tag,"trans:") == 0) {
      for (unsigned i = 0; i < v_trans.size(); i++) {
        for (unsigned j = 0; j < v_trans[i].size(); j++) {
          fscanf(fp,"%f",&f_val);
          v_trans[i][j] = f_val;
        }
      }
    }
    else if (strcmp(tag,"rtrans:") == 0) {
      for (unsigned i = 0; i < v_rtrans.size(); i++) {
        for (unsigned j = 0; j < v_rtrans[i].size(); j++) {
          fscanf(fp,"%f",&f_val);
          v_rtrans[i][j] = f_val;
        }
      }
    }
    else fprintf(stderr,"unknown tag ``%s''\n",tag);
  }
}/*}}}*/

void SaveHMMGMG(string filename, HMM_GMM &model)/*{{{*/
{
  vector<GaussianMixture*> &statePool = *(model.getpStatePool(USE));
  vector<Gaussian*> &gaussPool = *(model.getpGaussPool(USE));

  FILE *fp = FOPEN(filename.c_str(),"w");
  fprintf(fp,"HMMSet 1\n");
  model.SaveHMM(fp,ASCII);

  fprintf(fp,"StateSet %d\n", static_cast<int>(statePool.size()));
  for (unsigned i = 0; i < statePool.size(); i++)
    statePool[i]->SaveGaussianMixture(fp,ASCII);

  fprintf(fp,"GaussianSet %d\n",static_cast<int>(gaussPool.size()));
  for (unsigned i = 0; i < gaussPool.size(); i++)
    gaussPool[i]->SaveGaussian(fp,ASCII);
  fclose(fp);
}/*}}}*/

void LoadHMMGMG(/*{{{*/
    string filename,
    HMM_GMM *p_model,
    vector<GaussianMixture*> *statePool,
    vector<Gaussian*> *gaussPool
    )
{
  assert(p_model != 0);


  FILE *fp = FOPEN(filename.c_str(),"r");
  char tag[1024];
  int i_val;

  while (gettag(fp, tag)) {
    if (strcmp(tag,"HMMSet") == 0) {
      fscanf(fp,"%d",&i_val);
      if (i_val != 1) ErrorExit(__FILE__,__LINE__,-1,"LoadHMMGMG() can only read 1 HMM");
      p_model->LoadHMM(fp);
    }
    else if (strcmp(tag,"StateSet") == 0) {
      fscanf(fp,"%d",&i_val);
      statePool[0].resize(i_val);
      statePool[1].resize(i_val);
      for (int i = 0; i < i_val; i++) {
        statePool[0][i] = new GaussianMixture();
        statePool[0][i]->LoadGaussianMixture(fp);
        statePool[0][i]->setpGaussPool(&gaussPool[0]);
        statePool[1][i] = new GaussianMixture(statePool[0][i]->getDim(),statePool[0][i]->getNmix(), &gaussPool[1]);
        statePool[1][i]->copyGaussIdx(statePool[0][i]);
        statePool[1][i]->setpGaussPool(&gaussPool[1]);
      }
    }
    else if (strcmp(tag,"GaussianSet") == 0) {
      fscanf(fp,"%d",&i_val);
      gaussPool[0].resize(i_val);
      gaussPool[1].resize(i_val);
      for (int i = 0; i < i_val; i++) {
        gaussPool[0][i] = new Gaussian();
        gaussPool[0][i]->LoadGaussian(fp);
        gaussPool[1][i] = new Gaussian(gaussPool[0][i]->getDim());
      }
    }
    else fprintf(stderr,"unknown tag ``%s''\n",tag);
  }
  p_model->setpStatePool(&statePool[0],USE);
  p_model->setpStatePool(&statePool[1],UNUSE);
  p_model->setpGaussPool(&gaussPool[0],USE);
  p_model->setpGaussPool(&gaussPool[1],UNUSE);
  fclose(fp);

}/*}}}*/

void LoadHMMGMG(/*{{{*/
    string filename,
    HMM_GMM *p_model,
    vector<GaussianMixture*> &statePool,
    vector<Gaussian*> &gaussPool
    )
{
  assert(p_model != 0);


  FILE *fp = FOPEN(filename.c_str(),"r");
  char tag[1024];
  int i_val;

  while (gettag(fp, tag)) {
    if (strcmp(tag,"HMMSet") == 0) {
      fscanf(fp,"%d",&i_val);
      if (i_val != 1) ErrorExit(__FILE__,__LINE__,-1,"LoadHMMGMG() can only read 1 HMM");
      p_model->LoadHMM(fp);
    }
    else if (strcmp(tag,"StateSet") == 0) {
      fscanf(fp,"%d",&i_val);
      statePool.resize(i_val);
      for (int i = 0; i < i_val; i++) {
        statePool[i] = new GaussianMixture();
        statePool[i]->LoadGaussianMixture(fp);
        statePool[i]->setpGaussPool(&gaussPool);
      }
    }
    else if (strcmp(tag,"GaussianSet") == 0) {
      fscanf(fp,"%d",&i_val);
      gaussPool.resize(i_val);
      for (int i = 0; i < i_val; i++) {
        gaussPool[i] = new Gaussian();
        gaussPool[i]->LoadGaussian(fp);
      }
    }
    else fprintf(stderr,"unknown tag ``%s''\n",tag);
  }
  p_model->setpStatePool(&statePool,USE);
  p_model->setpGaussPool(&gaussPool,USE);
  fclose(fp);
}/*}}}*/

bool DeleteState(unsigned idx, HMM_GMM &model, set<int> &state_recycler, set<int> &gauss_recycler) {/*{{{*/
  cout << "Deleting state index " << idx << " in statePool\n";
  if (state_recycler.find(idx) != state_recycler.end()) {
    cerr << "Error: statePool[" << idx << "] is already in recycler\n";
    return false;
  }
  state_recycler.insert(idx);
  model.deleteStateIdx(idx);
  /* Check if the associated Gaussians are still in use */
  vector<GaussianMixture*> &statePool = *(model.getpStatePool(USE));
  GaussianMixture *state = statePool[idx];
  for (int x = 0; x < state->getNmix(); x++) {
    int gidx = state->getGaussIdx(x);
    bool notused = true;
    /* Check if gidx is still in use */
    for (unsigned i = 0; i < statePool.size(); i++) {
      if (i == idx) continue;
      if (statePool[i]->containGidx(gidx)) {
        notused = false;
        break;
      }
    }
    if (notused) gauss_recycler.insert(gidx);
  }
  return true;
}/*}}}*/

int GetGaussian(HMM_GMM &model, set<int> *p_gauss_recycler, int dim)/*{{{*/
{
  vector<Gaussian*> *pgaussPool[2] =
  { model.getpGaussPool(USE),model.getpGaussPool(UNUSE) };
  int gid;
  if (p_gauss_recycler == NULL || p_gauss_recycler->empty()) {
    gid = pgaussPool[0]->size();
    pgaussPool[0]->push_back(new Gaussian(dim));
    pgaussPool[1]->push_back(new Gaussian(dim));
  }
  else{
    gid = *p_gauss_recycler->begin();
    p_gauss_recycler->erase(p_gauss_recycler->begin());
  }

  // DEBUG/*{{{*/
  /*
     cout << "gauss_recycler = {";
     for (set<int>::iterator it = p_gauss_recycler->begin(); it != p_gauss_recycler->end(); it++)
     cout << "\t" << *it;
     cout << "}\n";
     */
  /*}}}*/
  //cout << "GetGaussian() = " << gid << endl;
  return gid;
}/*}}}*/

int GetState(HMM_GMM &model, set<int> *p_state_recycler, set<int> *p_gauss_recycler, const int dim, const int num_mix)/*{{{*/
{
  vector<GaussianMixture*> *pstatePool[2] =
  { model.getpStatePool(USE), model.getpStatePool(UNUSE) };
  vector<Gaussian*> *pgaussPool[2] =
  { model.getpGaussPool(USE), model.getpGaussPool(UNUSE) };
  assert(pstatePool[0] != NULL);
  assert(pstatePool[1] != NULL);
  assert(pgaussPool[0] != NULL);
  assert(pgaussPool[1] != NULL);
  assert(pstatePool[0]->size() == pstatePool[1]->size());
  assert(pgaussPool[0]->size() == pgaussPool[1]->size());
  int nsid;

  /* Nothing in recycler then create a new state */
  if (p_state_recycler == NULL || p_state_recycler->empty()) {
    nsid = pstatePool[0]->size();
    for (int u = 0; u < 2; u++) {
      pstatePool[u]->push_back(new GaussianMixture(dim,num_mix,pgaussPool[u]));
    }
  }
  else{
    nsid = *p_state_recycler->begin();
    p_state_recycler->erase(p_state_recycler->begin());
    for (int u = 0; u < 2; u++) {
      (*pstatePool[u])[nsid]->setNmix(num_mix);
    }
  }

  for (int x = 0; x < num_mix; x++) {
    int gidx = GetGaussian(model,p_gauss_recycler,dim);
    for (int u = 0; u < 2; u++) {
      (*pstatePool[u])[nsid]->setGaussIdx(x,gidx);
    }
  }
  //cout << "GetState() = " << nsid << endl;
  return nsid;
}/*}}}*/

int NewStateCopy(int sid, HMM_GMM &model, set<int> *p_state_recycler, set<int> *p_gauss_recycler) /*{{{*/
{

  GaussianMixture *state;
  /* state is the original USE state */
  state = model.getpStatePool(USE)->operator[](sid);

  int dim = state->getDim();
  int num_mix = state->getNmix();

  /* nsid is a state (newly created of gathered from state recycler) */
  int nsid = GetState(model,p_state_recycler,p_gauss_recycler,dim,num_mix);

  GaussianMixture *new_state =
    model.getpStatePool(USE)->operator[](nsid);

  /* Copy only USE state parameters */
  *new_state = *state;
  /* Copy only USE mixture parameters */
  for (int x = 0; x < num_mix; x++) {
    *(new_state->getpGauss(x)) = *(state->getpGauss(x));
  }

  return nsid;
}/*}}}*/

void RemoveTrash(HMM_GMM &model, vector<GaussianMixture*> statePool[2], vector<Gaussian*> gaussPool[2], set<int> &state_recycler, set<int> &gauss_recycler)/*{{{*/
{
  int idx;
  GaussianMixture *pstate;
  Gaussian *pgauss;
  while (!state_recycler.empty()) {
    /* Note that set<int> is sorted. Hence when deleting the last  *
     * index, then other to-be-deleted index is not changed.       */
    idx = *state_recycler.rbegin();
    cout << "Remove state index " << idx << endl;
    set<int>::iterator it = state_recycler.end(); it--;
    state_recycler.erase(it);
    for (int u = 0; u < 2; u++) {
      pstate = statePool[u][idx];
      delete pstate;
      statePool[u].erase(statePool[u].begin()+idx);
    }
    model.cancelStateIdx(idx);
  }
  while (!gauss_recycler.empty()) {
    idx = *gauss_recycler.rbegin();
    cout << "Remove gauss index " << idx << endl;
    set<int>::iterator it = gauss_recycler.end(); it--;
    gauss_recycler.erase(it);
    for (int u = 0; u < 2; u++) {
      pgauss = gaussPool[u][idx];
      delete pgauss;
      gaussPool[u].erase(gaussPool[u].begin()+idx);
      for (unsigned i = 0; i < statePool[u].size(); i++) {
        statePool[u][i]->cancelGaussIdx(idx);
      }
    }
  }
}/*}}}*/

