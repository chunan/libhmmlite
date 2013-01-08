#include "hmmlite.h"
#include "utility.h"
#include <cmath>
#include <iomanip>
#include <cstdio>
#include <cstring>
#include <set>
#include <algorithm>
#include <sstream>
#include <iterator>

using namespace std;
using namespace atlas;

const int REQUIRED_FILE = 3;


double GaussianMixture::REQUIRED_FRAME = 40;
double HMM_GMM::TRANS_ZERO = 0.01;
int    HMM_GMM::allowedNDel = 2;


template<class T>
void printVVec(vector<vector< T> >& vvec, const char *msg) {
  printf("==========printVVec(%s)=================\n",msg);
  for (unsigned i = 0; i < vvec.size(); ++i) {
    if (vvec[i].empty()) continue;
    printf("%3d: ", i);
    for (unsigned t = 0; t < vvec[i].size(); t++) {
      printf("%.2g\t",vvec[i][t]);
    }
    printf("\n");
  }
}

/**
 * @mainpage  Libhmmlit -- a lightweight HMM library
 *
 * Support training and testing with general HMM topology.
 */

void G_strip (char *theString) {/*{{{*/
  // Point to last character
  char *p = &theString [strlen (theString) - 1];
  // Move forward to first non-whitespace
  while (isspace (*p)) --p;
  // Move back one to beginning of whitespace
  // and terminate the string at that point
  *(++p) = '\0';
  // Point to front
  p = theString;
  // Find first non-whitespace
  while (isspace (*p)) ++p;
  // Copy from here (first non-whitespace) to beginning
  strcpy (theString, p);
}/*}}}*/

static int gettag(FILE *fd, char *tag) {/*{{{*/
  if (fscanf(fd, "%s", tag) != 1) return 0;
  G_strip(tag);
  return 1;
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

void GaussianMixture::ClearWeight(double val)/*{{{*/
{
  pthread_mutex_lock(&S_mutex);

  v_weight.assign(v_weight.size(), val);

  pthread_mutex_unlock(&S_mutex);
}/*}}}*/

bool GaussianMixture::normWeight(double & weightsum)/*{{{*/
{

  weightsum = 0.0;
  for (unsigned x = 0; x < v_weight.size(); x++)
    weightsum += v_weight[x];
  if (v_weight.size() > 1 && weightsum < getRequiredFrame() * v_weight.size()) {
    cerr << "GaussianMixture::normWeight(): State frame ("
         << weightsum
         << ") too small (< " 
         << getRequiredFrame() * v_weight.size()
         << "), mixture weights are not updated.\n";
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
  v_left = model.v_left;
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
    fprintf(fp," %2d",left[i]);
  fprintf(fp,"\n");

  fprintf(fp,"right:");
  for (unsigned i = 0; i < right.size(); i++)
    fprintf(fp," %2d",right[i]);
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
  return pStatePool[who]->at(v_state[s]);
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

void HMM_GMM::ClearPi(UseType u, double val) {/*{{{*/
  int who = (u == USE) ? use : 1-use;
  pthread_mutex_lock(&H_mutex);
  pi[who].assign(pi[who].size(), val);
  pthread_mutex_unlock(&H_mutex);
}/*}}}*/

void HMM_GMM::ClearTrans(double val) {/*{{{*/
  for (unsigned i = 0; i < v_trans.size(); i++) {
    v_trans[i].assign(v_trans[i].size(), val);
  }
  for (unsigned i = 0; i < v_rtrans.size(); i++) {
    v_rtrans[i].assign(v_rtrans[i].size(), val);
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
  ClearPi(UNUSE, 0.0);

  /* Trans */
  ClearIJ(0.0);

  /* state->weight */
  for (unsigned i = 0; i < pStatePool[unuse]->size(); i++)
    if (state_isUsed[i]) (*pStatePool[unuse])[i]->ClearWeight(0.0);

  /* Gaussian pool */
  vector<Gaussian *> &vGauss = *getpGaussPool(UNUSE);
  for (unsigned g = 0; g < vGauss.size(); g++) {
    if (!gauss_isUsed[g]) continue;
    vGauss[g]->ClearGaussian();
    vGauss[g]->setDiag(getpGaussPool(USE)->at(g)->getDiag());
  }

}/*}}}*/

void HMM_GMM::ClearIJ(double val) {/*{{{*/
  accum_ij.resize(i_nstate);
  for (unsigned i = 0; i < accum_ij.size(); i++) {
    accum_ij[i].assign(i_nstate, val);
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

void HMM_GMM::SyncUsed()/*{{{*/
{
  vector<Gaussian *> &vGauss = *getpGaussPool(USE);
  vector<GaussianMixture *> &vState = *getpStatePool(USE);

  state_isUsed.assign(vState.size(), false);
  gauss_isUsed.assign(vGauss.size(), false);


  for (int j = 0; j < i_nstate; j++) {
    state_isUsed[getGMidx(j)] = true;
    GaussianMixture *state  = getpGM(j,USE);
    int mixsize = state->getNmix();
    for (int x = 0; x < mixsize; x++)
      gauss_isUsed[state->getGaussIdx(x)] = true;
  }
}/*}}}*/

void HMM_GMM::CalLogBjOtPxs(int nframe) {/*{{{*/

  vector<GaussianMixture*>& gmpool = *pStatePool[use];
  bjOt.resize(gmpool.size());
  px_s.resize(gmpool.size());

  /* For each state */
  for (unsigned j = 0; j < gmpool.size(); j++) {
    if (!state_isUsed[j]) continue;
    int mixsize = gmpool[j]->getNmix();
    bjOt[j].resize(nframe);
    px_s[j].resize(mixsize);

    if (mixsize == 1) { // single-Gaussian
      px_s[j][0].resize(nframe);
      for (int t = 0; t < nframe; t++) {
        bjOt[j][t] = bgOt[gmpool[j]->getGaussIdx(0)][t];
        px_s[j][0][t] = 0.0; // = logp(x| ot, j) = log(1)
      }

    } else { // multi-Gaussians
      vector<double> weight(gmpool[j]->getWeight());
      for_each(weight.begin(), weight.end(), LOG);
      /* For each x: px_s = logp(ot, x| j)*/
      for (int x = 0; x < mixsize; x++) {
        px_s[j][x].resize(nframe);
        for (int t = 0; t < nframe; t++)
          px_s[j][x][t] = LProd(weight[x], bgOt[gmpool[j]->getGaussIdx(x)][t]);
      }
      /* For each frame, bjOt[j][t] = px_s[j][SUM x][t] = log(ot| j) */
      for (int t = 0; t < nframe; t++) {
        bjOt[j][t] = px_s[j][0][t];
        /* sum over x */
        for (int x = 1; x < mixsize; x++)
          bjOt[j][t] = LAdd(bjOt[j][t], px_s[j][x][t]);
        /* normalize px_s[j][x][t] / px_s[][SUM x][t] */
        for (int x = 0; x < mixsize; x++)
          px_s[j][x][t] = LDiv(px_s[j][x][t], bjOt[j][t]);
      }
      /* For each x: px_s = logp(x|ot, j) */
    }

    /* px_s = log p(o, g | j) */
  }
  /* Recycle bgOt because it is memory wasted */
  bgOt.clear();
}/*}}}*/

void HMM_GMM::CalLogBjOt(int nframe) {/*{{{*/

  vector<GaussianMixture*>& gmpool = *pStatePool[use];
  bjOt.resize(gmpool.size());

  /* For each state */
  for (unsigned j = 0; j < gmpool.size(); j++) {
    if (!state_isUsed[j]) continue;

    int mixsize = gmpool[j]->getNmix();
    bjOt[j].resize(nframe);

    if (mixsize == 1) { // single-Gaussian
      for (int t = 0; t < nframe; t++)
        bjOt[j][t] = bgOt[gmpool[j]->getGaussIdx(0)][t];

    } else { // multi-Gaussians
      vector<double> weight(gmpool[j]->getWeight());
      for_each(weight.begin(), weight.end(), LOG);
      bjOt[j].assign(nframe, LZERO);
      for (int t = 0; t < nframe; t++) {/* For each frame */
        for (int x = 0; x < mixsize; x++) {/* For each Gaussian */
          int g = gmpool[j]->getGaussIdx(x);
          bjOt[j][t] = LAdd(bjOt[j][t], LProd(weight[x], bgOt[g][t]));
        } /* for each Gaussian */
      } /* for each frame */
    }

  }
  /* Recycle bgOt because it is memory wasted */
  bgOt.clear();
}/*}}}*/

void HMM_GMM::CalLogAlpha(int nframe, vector<int> *p_label) {/*{{{*/
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
      alpha[i][0] = LProd(pi[use][sno],  bjOt[v_state[sno]][0]);
    }
  }

  /* Fill the table */
  for (int t = 1; t < nframe; t++) {
    for (unsigned j = 0; j < alpha.size(); j++) {
      alpha[j][t] = LZERO;

      /* Transitted from (i_start ~ i_end) -> sno_j */
      int i_start = useLabel ? max(0, static_cast<int>(j-allowedNDel-1)) : 0;
      int i_end   = useLabel ? j+1 : i_nstate;
      int sno_j   = useLabel ? (*p_label)[j] : j;

      for (int i = i_start; i < i_end; i++) {
        int sno_i = useLabel ? (*p_label)[i] : i;
        if (v_trans[sno_i][sno_j] <= LSMALL) continue;
        alpha[j][t] = LAdd(alpha[j][t],
                           LProd(alpha[i][t-1], v_trans[sno_i][sno_j]));
      }

      alpha[j][t] = LProd(alpha[j][t], bjOt[v_state[sno_j]][t]);
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

void HMM_GMM::CalLogBeta(int nframe, vector<int> *p_label) {/*{{{*/
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
  vector<double> nextbeta(i_nstate);

  /* Fill the table */
  for (int t = nframe-2; t >= 0; t--) {
    /* Calculate nextbeta */
    for (unsigned j = 0; j < beta.size(); j++) {
      int sno_j = useLabel ? (*p_label)[j] : j;
      nextbeta[j] = LProd(beta[j][t+1], bjOt[v_state[sno_j]][t+1]);
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
    fprintf(stderr,"HMM_GMM::CalLogPrO: LogProb = zero\n");
    printVVec(bgOt,"bgOt");
    printVVec(bjOt,"bjOt");
    printVVec(alpha,"alpha");
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

void HMM_GMM::CalLogEpsilon(int nframe, vector<int> *p_label) {/*{{{*/
  bool useLabel = (p_label != NULL);
  epsilon.resize(i_nstate);
  for (unsigned i = 0; i < epsilon.size(); i++) {
    epsilon[i].resize(i_nstate);
    for (unsigned j = 0; j < epsilon[i].size(); j++)
      epsilon[i][j].resize(nframe - 1);
  }

  for (int t = 0; t < nframe - 1; t++) {
    for (unsigned j = 0; j < epsilon.size(); j++) {
      int sno_j = useLabel ? (*p_label)[j] : j;
      double bjot_beta_prO = LDiv(
          LProd(bjOt[v_state[sno_j]][t+1], beta[j][t+1]), prO);
      int i_start = useLabel ? max(0u,j-allowedNDel-1) : 0;
      int i_end   = useLabel ? j+1 : i_nstate;
      for (int i = i_start; i < i_end; i++) {
        int sno_i = useLabel ? (*p_label)[i] : i;

        if (v_trans[sno_i][sno_j] < LSMALL) {
          epsilon[i][j][t] = LZERO;
        } else {
          epsilon[i][j][t] = LProd(
              LProd(alpha[i][t], v_trans[sno_i][sno_j]), bjot_beta_prO);
        }

      } /* for i */
    } /* for j */
  } /* for t */
  /* epsilon = logp(O, qt: i->j) */
}/*}}}*/

void HMM_GMM::CalLogEpsilonBound(int nframe, vector<int> *p_endf) {/*{{{*/
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

void HMM_GMM::AccumPi(vector<int> *p_label, double obs_weight) {/*{{{*/
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

void HMM_GMM::AccumIJ(int nframe, vector<int> *p_label, double obs_weight) {/*{{{*/
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
      for (int t = 0; t < nframe_1; t++) {
        accum_ij[sno_i][sno_j] += obs_weight * epsilon[i][j][t];
      }
    }
  }
  pthread_mutex_unlock(&H_mutex);
}/*}}}*/

void HMM_GMM::AccumWeightGaussian(float **obs, int nframe, int dim, /*{{{*/
                                  UpdateType udtype, vector<int> *p_label,
                                  double obs_weight) {
  bool useLabel = (p_label != NULL);

  /* For each state */
  for (unsigned i = 0; i < gamma.size(); i++) {
    int sno_i = useLabel ? (*p_label)[i] : i;
    GaussianMixture *state = getpGM(sno_i,UNUSE);
    /* For each gaussian associated with this state */
    for (int x = 0; x < state->getNmix(); x++) {
      Gaussian *pg = state->getpGauss(x);
      for (int t = 0; t < nframe; t++) {
        double gamma_i_x_t = obs_weight * gamma[sno_i][t] * px_s[sno_i][x][t];
        if (gamma_i_x_t <= ZERO) continue;
        pg->AddData(obs[t],dim,gamma_i_x_t,udtype);
        state->setWeight(x, gamma_i_x_t, ADD);
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

void HMM_GMM::normTrans() {/*{{{*/
  for (int i = 0; i < i_nstate; i++) {
    double sum = 0.0;
    /* Calculate nonZERO transition sum */
    for (int j = 0; j < i_nstate; j++) {
      if (v_trans[i][j] < ZERO) v_trans[i][j] = 0.0;
      else sum += v_trans[i][j];
    }
    double trans_floor = (sum - v_trans[i][i]) * TRANS_ZERO / i_nstate;
    sum = 0.0;

    /* Calculate transitions above trans_floor */
    for (int j = 0; j < i_nstate; j++) {
      if (v_trans[i][j] < ZERO) continue;
      if (v_trans[i][j] < trans_floor) {
        v_trans[i][j] = 0.0;
      }
      else sum += v_trans[i][j];
    }
    if (sum < ZERO) {
      cerr << "HMM_GMM::normTrans(): no occupation for state no " << i
        << ", transition set to zero" << endl;

      for (int j = 0; j < i_nstate; j++) {
        cout << "v_trans[" << i << "][" << j << "] = " << v_trans[i][j] << endl;
        v_trans[i][j] = 0.0;
      }
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

void HMM_GMM::normTransOther(int sno) {/*{{{*/
  assert(sno >= 0 && sno < i_nstate);
  float fac = 0.0f;
  for (int j = 0; j < i_nstate; ++j) {
    if (j != sno && v_trans[sno][j] > 0.0f)
      fac += v_trans[sno][j];
  }

  fac = (1.0f - v_trans[sno][sno]) / fac;
  for (int j = 0; j < i_nstate; ++j) {
    if (j != sno && v_trans[sno][j] > 0.0f)
      v_trans[sno][j] *= fac;
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

void HMM_GMM::accum2Trans() {/*{{{*/

  for (int i = 0; i < i_nstate; i++) {

    /* Accumulate transition from state i */
    double sum = 0.0;
    for (int j = 0; j < i_nstate; j++)
      if (accum_ij[i][j] > ZERO) sum += accum_ij[i][j];

    /* Truncate small transitions */
    double trans_floor = (sum - accum_ij[i][i]) * TRANS_ZERO / i_nstate;
    sum = 0.0;
    for (int j = 0; j < i_nstate; j++) {
      if (accum_ij[i][j] > trans_floor) sum += accum_ij[i][j];
    }

    if (sum < ZERO) {
      cerr << "Warning: accum2Trans() occupation[state " << i
        << "] too small " << endl;
      v_trans[i].assign(i_nstate, 0.0);
      v_trans[i][i] = 1.0;
    } else {
      /* Normalize transitions */
      for (int j = 0; j < i_nstate; j++) {
        if (accum_ij[i][j] > trans_floor)
          v_trans[i][j] = accum_ij[i][j] / sum;
        else v_trans[i][j] = 0.0;
      }
    }
  }

#if 0
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
#endif

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

void HMM_GMM::EMUpdate(set<int> *p_delete_list,/*{{{*/
                       double backoff_weight,
                       UpdateType udtype) {
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

    // 2. The rest is taken care of by normMeanCov()
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
  SyncLeft();

}/*}}}*/


double HMM_GMM::EMObs(float **obs, int nframe, int dim, double obs_weight, UpdateType udtype) /*{{{*/
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

double HMM_GMM::EMObsLabel(float **obs, int nframe, int dim, vector<int> *p_label, double obs_weight, UpdateType udtype) /*{{{*/
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

double HMM_GMM::EMObsBound(float **obs, int nframe, int dim, Labfile *p_reflabfile, double obs_weight, UpdateType udtype) /*{{{*/
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

double HMM_GMM::CalLogDelta(vector<int> &state_seq, /*{{{*/
                            vector<float>* likelihood_seq,
                            const vector<int> *p_endf) {
  cout << "CalLogDelta()" << endl;
  bool bndConstraint = (p_endf != NULL);
  vector<vector<int> > path;
  int nframe = bjOt[0].size();
  int nframe_1 = nframe - 1;
  /* Allocate memory */
  delta.resize(i_nstate);
  path.resize(i_nstate);
  for (int j = 0; j < i_nstate; j++) {
    delta[j].resize(nframe);
    path[j].resize(nframe);
    delta[j][0] = LProd(pi[use][j], bjOt[v_state[j]][0]);
  }
  cout << "Fill table..." << std::flush;
  /* Fill the table */
  int i_end = 0;
  bool isBound = true;
  int left_i;
  double newscore;
  for (int t = 1; t < nframe; t++) {
    if (bndConstraint) {
      if ((isBound = (t-1 == (*p_endf)[i_end]))) i_end++;
    }
    for (int j = 0; j < i_nstate; j++) {
      // default: self-transition
      delta[j][t] = LProd(delta[j][t-1], v_trans[j][j]);
      path[j][t] = j;

      if (!isBound) {// inside segment: only left-to-right on neucli states
        if ((left_i = getLeft(j)) != -1) {
          newscore = LProd(delta[left_i][t-1], v_trans[left_i][j]);
          if (newscore > delta[j][t]) {
            delta[j][t] = newscore;
            path[j][t] = left_i;
          }
        }

      } else {// between segments: left-to-right | boundary trans
        if ((left_i = getLeft(j)) != -1) { // left-to-right
          newscore = LProd(delta[left_i][t-1], v_trans[left_i][j]);
          if (newscore > delta[j][t]) {
            delta[j][t] = newscore;
            path[j][t] = left_i;
          }
        } else { // boundary trans
          for (int i = 0; i < i_nstate; i++) {
            if (v_trans[i][j] <= LSMALL) continue;
            newscore = LProd(delta[i][t-1], v_trans[i][j]);
            if (newscore > delta[j][t]) {
              delta[j][t] = newscore;
              path[j][t] = i;
            }
          }
        }

      } /* if isBound, else */
      delta[j][t] = LProd(delta[j][t], bjOt[v_state[j]][t]);
    }
  }
  cout << "done\nBacktrace..." << std::flush;
  /**** backtrace ****/
  state_seq.resize(nframe);
  if (likelihood_seq) likelihood_seq->resize(nframe);
  double BestScore = LZERO;

  /* find optimal last state */
  for (int i = i_nstate - 1; i >= 0; i--) {
    if (getRight(i) == -1 && delta[i][nframe_1] > BestScore) {
      BestScore = delta[i][nframe_1];
      state_seq[nframe_1] = i;
    }
  }

  /* trace back */
  if (likelihood_seq) (*likelihood_seq)[nframe_1] = BestScore;
  for (int t = nframe - 2; t >= 0; t--) {
    state_seq[t] = path[state_seq[t+1]][t+1];
    if (likelihood_seq) (*likelihood_seq)[t] = delta[state_seq[t]][t];
  }
  /* HMM state index -> GM index */
  /*
  for (int t = 0; t < nframe; ++t) {
    state_seq[t] = v_state[state_seq[t]];
  }
  */
  cout << "done\nEnd" << endl;
  return BestScore;
}/*}}}*/

double HMM_GMM::CalLogDelta(vector<int> &state_seq, /*{{{*/
                            vector<float>* likelihood_seq) {
  vector<vector<int> > path;
  int nframe = bjOt[v_state[0]].size();
  //assert(nframe > 0);
  /* Allocate memory */
  path.resize(i_nstate);
  delta.resize(i_nstate);
  for (unsigned j = 0; j < path.size(); j++) {
    path[j].resize(nframe);
    delta[j].resize(nframe);
    delta[j][0] = LProd(pi[use][j], bjOt[v_state[j]][0]);
  }
  /* Fill the table */
  int left_i;
  double newscore;
  for (int t = 1; t < nframe; t++) {
    for (int j = 0; j < i_nstate; j++) {
      // default: self-transition
      //assert(j < delta.size());
      //assert(t < delta[j].size());
      delta[j][t] = LProd(delta[j][t-1], v_trans[j][j]);
      path[j][t] = j;

      //assert(j < v_left.size());
      for (unsigned k = 0; k < v_left[j].size(); ++k) {
        left_i = v_left[j][k];
        newscore = LProd(delta[left_i][t-1], v_trans[left_i][j]);
        if (newscore > delta[j][t]) {
          delta[j][t] = newscore;
          path[j][t] = left_i;
        }
      }

      delta[j][t] = LProd(delta[j][t], bjOt[v_state[j]][t]);
    }
  }
  /**** backtrace ****/
  state_seq.resize(nframe);
  if (likelihood_seq) likelihood_seq->resize(nframe);
  double BestScore = LZERO;

  /* find optimal last state */
  int nframe_1 = nframe - 1;
  for (int i = i_nstate - 1; i >= 0; i--) {
    if (getRight(i) == -1 && delta[i][nframe_1] > BestScore) {
      BestScore = delta[i][nframe_1];
      state_seq[nframe_1] = i;
    }
  }

  /* trace back */
  if (likelihood_seq) (*likelihood_seq)[nframe_1] = BestScore;
  for (int t = nframe - 2; t >= 0; t--) {
    state_seq[t] = path[state_seq[t+1]][t+1];
    if (likelihood_seq) (*likelihood_seq)[t] = delta[state_seq[t]][t];
  }
  return BestScore;
}/*}}}*/


/****** I/O ******/
void GaussianMixture::SaveGaussianMixture(FILE *fp, const DataType type)/*{{{*/
{
  if (type == BINARY)
    fprintf(stderr,"GaussianMixture::SaveGaussianMixture(): does not support binary\n");
  display(fp);
}/*}}}*/

void GaussianMixture::LoadGaussianMixture(FILE *fp) {/*{{{*/
  char buff[1024];
  fscanf(fp,"%s",buff);
  assert(strcmp(buff,"GaussianMixture") == 0);
  fscanf(fp,"%s",buff);
  if (strcmp(buff,"ascii") == 0) {
    ReadAscii(fp);
  } else {
    fprintf(stderr,"Unknown tag for GaussianMixture, only ascii/binary allowed\n");
  }
}/*}}}*/

void GaussianMixture::LoadGaussianMixture(ifstream& ifs) {/*{{{*/

  string line, tag;
  Getline(ifs, line);
  stringstream iss(line);

  iss >> tag;
  assert(tag.compare("GaussianMixture") == 0);

  iss >> tag;
  if (tag.compare("ascii") == 0) {
    ReadAscii(ifs);
  } else {
    cerr << "Unknown tag: ``" << tag << "'' for GaussianMixture,"
      << " only ``ascii'' allowed\n";
  }

}/*}}}*/

void GaussianMixture::ReadAscii(ifstream& ifs) {/*{{{*/

  string line, tag;
  int i_val;

  while (Getline(ifs, line)) {
    stringstream iss(line);
    iss >> tag;

    if (tag.compare("EndGaussianMixture") == 0) {
      break;

    } else if (tag.compare("name:") == 0) {
      iss >> s_name;

    } else if (tag.compare("dim:") == 0) {
      iss >> dim;

    } else if (tag.compare("nmix:") == 0) {
      iss >> i_val;
      setNmix(i_val);

    } else if (tag.compare("weight:") == 0) {
      for (unsigned x = 0; x < v_weight.size(); x++)
        iss >> v_weight[x];

    } else if (tag.compare("gaussidx:") == 0) {
      for (unsigned x = 0; x < v_gaussidx.size(); x++)
        iss >> v_gaussidx[x];

    } else {
      ErrorExit(__FILE__, __LINE__, 1, "unknown tag ``%s''\n", tag.c_str());

    }
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
    else {
      ErrorExit(__FILE__, __LINE__, 1, "unknown tag ``%s''\n", tag);
    }
  }
}/*}}}*/

void HMM_GMM::SaveHMM(FILE *fp, const DataType type)/*{{{*/
{
  if (type == BINARY)
    fprintf(stderr,"HMM_GMM::SaveHMM(): does not support binary\n");
  display(fp);
}/*}}}*/

void HMM_GMM::LoadHMM(FILE *fp) {/*{{{*/
  char buff[1024];
  fscanf(fp,"%s",buff);
  assert(strcmp(buff,"HMM") == 0);
  fscanf(fp,"%s",buff);
  if (strcmp(buff,"ascii")==0) {
    ReadAscii(fp);
  } else {
    fprintf(stderr,"Unknown tag for HMM, only ascii allowed\n");
  }
}/*}}}*/

void HMM_GMM::LoadHMM(ifstream& ifs) {/*{{{*/

  string line, tag;
  Getline(ifs, line);
  stringstream iss(line);

  iss >> tag;
  assert(tag.compare("HMM") == 0);

  iss >> tag;
  if (tag.compare("ascii") == 0) {
    ReadAscii(ifs);
  } else {
    cerr << "Unknown tag ``" << tag
      << "'' for HMM, only ascii/binary allowed\n";
  }

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
    else if (strcmp(tag,"state:") == 0 || strcmp(tag, "gmidx:") == 0) {
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
    else if (strcmp(tag,"pi:") == 0 || strcmp(tag,"pi dense:") == 0) {
      for (unsigned i = 0; i < pi[use].size(); i++) {
        fscanf(fp,"%f",&f_val);
        pi[use][i] = f_val;
      }
    }
    else if (strcmp(tag,"pi sparse:") == 0) {

      for (unsigned i = 0; i < pi[use].size(); i++) {
        fscanf(fp,"%d:%f", &i_val, &f_val);
        pi[use][i_val] = f_val;
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
    else {
      ErrorExit(__FILE__, __LINE__, 1, "unknown tag ``%s''\n",tag);
    }
  }
}/*}}}*/

void HMM_GMM::ReadAscii(ifstream& ifs) {/*{{{*/

  string line, tag;
  int i_val;
  float f_val;

  while (Getline(ifs, line)) {
    stringstream iss(line);
    iss >> tag;
    if (tag.compare("EndHMM") == 0) {
      break;

    } else if (tag.compare("pdf_weight:") == 0) {
      iss >> pdf_weight;

    } else if (tag.compare("nstate:") == 0) {
      iss >> i_val;
      setNstate(i_val);

    } else if (tag.compare( "gmidx:") == 0 || tag.compare("state:") == 0) {
      for (unsigned i = 0; i < v_state.size(); i++)
        iss >> v_state[i];

    } else if (tag.compare("left:") == 0) { /* Will be derived from transp */
      for (unsigned i = 0; i < left.size(); i++)
        iss >> left[i];

    } else if (tag.compare("right:") == 0) { /* Will be derived from transp */
      for (unsigned i = 0; i < left.size(); i++)
        iss >> right[i];

    } else if (tag.compare("pi") == 0) {
      iss >> tag;
      if (tag.compare("dense:") == 0) {
        for (unsigned i = 0; i < pi[use].size(); i++) iss >> pi[use][i];
      } else if (tag.compare("sparse:") == 0) {
        pi[use].assign(i_nstate, 0.0);
        while (iss >> tag) { // tokens of pattern i:f
          vector<string> val_pair = split(tag, ":");
          i_val = atoi(val_pair[0].c_str());
          f_val = atof(val_pair[1].c_str());
          pi[use][i_val] = f_val;
        }
      } else {
        cerr << "Unknown flag \"pi " << tag << "\"\n";
      } /* if dense, sparse, else */

    } else if (tag.compare("trans") == 0) {
      iss >> tag;
      if (tag.compare("dense:") == 0) {
        for (unsigned i = 0; i < v_trans.size(); i++) {
          Getline(ifs, line);
          iss.str(line);
          iss.clear();
          v_trans[i].assign(istream_iterator<double>(iss),
                            istream_iterator<double>());
        }
      } else if (tag.compare("sparse:") == 0) {
        for (unsigned i = 0; i < v_trans.size(); i++) {
          v_trans[i].assign(i_nstate, 0.0);
          Getline(ifs, line);
          iss.str(line);
          iss.clear();
          while (iss >> tag) { // tokens of pattern i:f
            vector<string> val_pair = split(tag, ":");
            i_val = atoi(val_pair[0].c_str());
            f_val = atof(val_pair[1].c_str());
            v_trans[i][i_val] = f_val;
          }
        }
      } /* if dense, else */
      SyncLeft();

    } else {
      ErrorExit(__FILE__, __LINE__, 1, "unknown tag ``%s''\n", tag.c_str());
    }
  }
}/*}}}*/

void HMM_GMM::SyncLeft() {/*{{{*/
  v_left.assign(i_nstate, vector<int>());
  for (int j = 0; j < i_nstate; ++j) {
    for (int i = 0; i < i_nstate; ++i) {
      if (i == j) continue;
      if (v_trans[i][j] > ZERO) v_left[j].push_back(i);
    }
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

#if 0
/* statePool[2], gaussPool[2] */
void LoadHMMGMG(/*{{{*/
    string filename,
    HMM_GMM *p_model,
    vector<GaussianMixture*> *statePool,
    vector<Gaussian*> *gaussPool) {

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
    else { }
  }
  p_model->setpStatePool(&statePool[0],USE);
  p_model->setpStatePool(&statePool[1],UNUSE);
  p_model->setpGaussPool(&gaussPool[0],USE);
  p_model->setpGaussPool(&gaussPool[1],UNUSE);
  fclose(fp);

}/*}}}*/
#endif

/* statePool[2], gaussPool[2] */
void LoadHMMGMG(/*{{{*/
    string filename,
    HMM_GMM *p_model,
    vector<GaussianMixture*> *statePool,
    vector<Gaussian*> *gaussPool) {

  assert(p_model != 0);

  ifstream ifs(filename.c_str(), ifstream::in);
  if (!ifs.good()) {
    ErrorExit(__FILE__, __LINE__, -1,
              "Unable to open file %s with flag ifstream::in\n",
              filename.c_str());
  }

  string tag, line;
  int i_val;
  while (Getline(ifs, line)) {
    if (line.empty()) continue;

    stringstream ss(line);
    ss >> tag;

    if (tag.compare("HMMSet") == 0) {
      ss >> i_val;
      if (i_val != 1)
        ErrorExit(__FILE__,__LINE__,-1,"LoadHMMGMG() can only read 1 HMM");
      p_model->LoadHMM(ifs);

    } else if (tag.compare("StateSet") == 0 ||
               tag.compare("GaussianMixtureSet") == 0) {
      ss >> i_val;
      statePool[0].resize(i_val);
      statePool[1].resize(i_val);
      for (int i = 0; i < i_val; i++) {
        statePool[0][i] = new GaussianMixture();
        statePool[0][i]->LoadGaussianMixture(ifs);
        statePool[0][i]->setpGaussPool(&gaussPool[0]);
        statePool[1][i] = new GaussianMixture(statePool[0][i]->getDim(),
                                              statePool[0][i]->getNmix(),
                                              &gaussPool[1]);
        statePool[1][i]->copyGaussIdx(statePool[0][i]);
        statePool[1][i]->setpGaussPool(&gaussPool[1]);
      }

    } else if (tag.compare("GaussianSet") == 0) {
      ss >> i_val;
      gaussPool[0].resize(i_val);
      gaussPool[1].resize(i_val);
      for (int i = 0; i < i_val; i++) {
        gaussPool[0][i] = new Gaussian();
        gaussPool[0][i]->LoadGaussian(ifs);
        gaussPool[1][i] = new Gaussian(gaussPool[0][i]->getDim());
      }

    } else {
      ErrorExit(__FILE__, __LINE__, -1, "Unknown tag: ``%s''\n", tag.c_str());
    }
  }
  p_model->setpStatePool(&statePool[0],USE);
  p_model->setpStatePool(&statePool[1],UNUSE);
  p_model->setpGaussPool(&gaussPool[0],USE);
  p_model->setpGaussPool(&gaussPool[1],UNUSE);
  ifs.close();

}/*}}}*/

#if 0
void LoadHMMGMG(/*{{{*/
    string filename,
    HMM_GMM *p_model,
    vector<GaussianMixture*> &statePool,
    vector<Gaussian*> &gaussPool
    ) {
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
    else {
      ErrorExit(__FILE__, __LINE__, 1, "unknown tag ``%s''\n", tag);
    }
  }
  p_model->setpStatePool(&statePool,USE);
  p_model->setpGaussPool(&gaussPool,USE);
  fclose(fp);
}/*}}}*/
#endif

void LoadHMMGMG(/*{{{*/
    string filename,
    HMM_GMM *p_model,
    vector<GaussianMixture*> &statePool,
    vector<Gaussian*> &gaussPool) {

  assert(p_model != 0);

  ifstream ifs(filename.c_str(), ios::in);
  if (!ifs.good()) {
    ErrorExit(__FILE__, __LINE__, -1,
              "Unable to open file %s with flag ifstream::in\n",
              filename.c_str());
  }

  string tag, line;
  int i_val;
  while (Getline(ifs, line)) {
    if (line.empty()) continue;

    stringstream ss(line);
    ss >> tag;

    if (tag.compare("HMMSet") == 0) {
      ss >> i_val;
      if (i_val != 1)
        ErrorExit(__FILE__,__LINE__,-1,"LoadHMMGMG() can only read 1 HMM");
      p_model->LoadHMM(ifs);

    } else if (tag.compare("StateSet") == 0 ||
               tag.compare("GaussianMixtureSet") == 0) {
      ss >> i_val;
      statePool.resize(i_val);
      for (int i = 0; i < i_val; i++) {
        statePool[i] = new GaussianMixture();
        statePool[i]->LoadGaussianMixture(ifs);
        statePool[i]->setpGaussPool(&gaussPool);
      }
    }
    else if (tag.compare("GaussianSet") == 0) {
      ss >> i_val;
      gaussPool.resize(i_val);
      for (int i = 0; i < i_val; i++) {
        gaussPool[i] = new Gaussian();
        gaussPool[i]->LoadGaussian(ifs);
      }

    } else {
      ErrorExit(__FILE__, __LINE__, 1, "unknown tag ``%s''\n", tag.c_str());
    }
  }
  p_model->setpStatePool(&statePool,USE);
  p_model->setpGaussPool(&gaussPool,USE);
  ifs.close();
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

