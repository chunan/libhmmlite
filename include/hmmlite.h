#ifndef __HMM_GAUSSIAN_H__
#define __HMM_GAUSSIAN_H__

#include <cstdlib>
#include <fstream>
#include <vector>
#include <deque>
#include <cassert>
#include <string>
#include <limits>
#include <set>
#include <pthread.h>
#include <cmath>
#include <algorithm>
#include "utility.h"
#include "libatlas_wrapper.h"

#define float_inf std::numeric_limits<float>::infinity()


enum UseType { USE, UNUSE};
enum SetType { ADD, SET};
enum DataType{ ASCII, BINARY};
enum UpdateType { UpdateAll, UpdateMean, UpdateCov };

using namespace std;
using namespace atlas;

class GaussianMixture { /*{{{*/
  public:
    GaussianMixture() {Init(0,0,0);}
    GaussianMixture(int d, int m, vector<Gaussian *> *p) {Init(d,m,p);}
    ~GaussianMixture() {}
    void Init(int d, int x, vector<Gaussian *> *p);
    const GaussianMixture & operator=(const GaussianMixture & gm);
    // Get value
    int getDim() const { return dim; }
    int getGaussIdx(unsigned x) const { assert(x < v_gaussidx.size()); return v_gaussidx[x]; }
    double getWeight(unsigned x) const { assert(x < v_gaussidx.size()); return v_weight[x]; }
    const vector<double>& getWeight() const { return v_weight; }
    Gaussian *getpGauss(unsigned x) const { assert(x < v_gaussidx.size()); return (*pGaussPool)[getGaussIdx(x)]; }
    vector<Gaussian *> *getpGaussPool() const { return pGaussPool; }
    int getNmix() const { return v_gaussidx.size(); }
    string getName() const { return s_name;}
    bool containGidx(int gidx);
    static double getRequiredFrame() { return REQUIRED_FRAME; }
    // Set value
    void setDim(int d) { dim = d; }
    void setNmix(int x) { v_weight.resize(x); v_gaussidx.resize(x); }
    void setGaussIdx(unsigned x, unsigned idx);
    void copyGaussIdx(const GaussianMixture *pstate);
    void setpGaussPool(vector<Gaussian *> *ptr) { pGaussPool = ptr;}
    void setWeight(int x, double val, SetType s);
    void copyWeight(GaussianMixture &s);
    void UniformWeight();
    bool cancelGaussIdx(int idx);
    void setName(const char *str) { s_name = str; }
    void setName(const string str) { s_name = str; }
    static void setRequiredFrame(double r) { REQUIRED_FRAME = r; }
    // Clear value
    void ClearWeight(double val);
    // Normalization
    bool normWeight(double &weightsum);
    void display(FILE *fp) const;
    void SaveGaussianMixture(FILE* fp, const DataType type = ASCII);
    void LoadGaussianMixture(FILE* fp);
    void LoadGaussianMixture(ifstream& ifs);
  private:
    void ReadAscii(FILE *fp);
    void ReadAscii(ifstream& ifs);

    /* Data */
    int dim;
    static double REQUIRED_FRAME;
    string s_name;
    vector<double> v_weight;
    vector<int> v_gaussidx;
    vector<Gaussian *> *pGaussPool;
    pthread_mutex_t S_mutex;
};/*}}}*/

class HMM_GMM /*{{{*/
{
  public:
    HMM_GMM() {Init();}
    ~HMM_GMM() {};
    /************ Get value **************/
    /* Return number of state */
    int getNstate() const { return i_nstate;}
    /* Return use */
    int getUse() const {return use;}
    /* Return s-th state index */
    int getGMidx(int s) const ;
    /* Return pointer to s-th state (use) */
    GaussianMixture *getpGM(int s, UseType u = USE) const ;
    /* Return Pi(s) (use) */
    double getPi(int s, UseType u = USE) const ;
    /* Return Trans(i,j) */
    double getTrans(int i, int j) const ;
    /* Return rTrans(i,j) */
    double getRTrans(int i, int j) const ;
    /* Return pointer to state pool */
    vector<GaussianMixture *> *getpStatePool(UseType u = USE) const ;
    vector<Gaussian *> *getpGaussPool(UseType u = USE) const ;
    /* Return name */
    string getName() const { return s_name;}
    /* Return occupation */
    double getOccupation(int i) const { return occupation[i]; }
    /* Return TRANS_ZERO */
    static double getTRANS_ZERO() { return TRANS_ZERO; }
    static int getAllowedNDel() { return allowedNDel; }
    /* Return left/right */
    int getLeft(int s) const { return left[s]; }
    int getRight(int s) const { return right[s]; }
    /* Return state/gauss_isUsed */
    bool getGisUsed(int g) const { return gauss_isUsed[g]; }
    bool getSisUsed(int sid) const { return state_isUsed[sid]; }
    double get_pdf_weight() const { return pdf_weight; }
    double getPrO() const { return prO; }
    const vector<int>& getVstate() const { return v_state; }
    /* Dump all EM-related variables */
    void dump_param() const;
    void display(FILE *fp) const;

    /************ Set value **************/
    /* Set number of state */
    void setNstate(int n);
    /* Set use */
    void setUse(int u) { assert(u==0 || u==1); use = u;}
    /* Set s-th state index */
    void setState(unsigned s, unsigned idx);
    /* Set s-th pi value (unused) */
    void setPi(int s, double val, UseType u = USE);
    void addPi(int s, double val, UseType u = USE);
    void UniformPi();
    /* Set trans(i,j) value (unused) */
    void setTrans(int i, int j, double val);
    void addTrans(int i, int j, double val);
    void copyTrans(int f, int t) { v_trans[t] = v_trans[f]; }
    void copyRTrans(int f, int t) { v_rtrans[t] = v_rtrans[f]; }
    /* Set left[i]/right[i] value */
    void setLeft(int s, int L);
    void setRight(int s, int R);
    void set_pdf_weight(double w) { pdf_weight = w; }
    //void updateLR();
    /* Set u(=0|1)'s state pool pointer */
    void setpStatePool(vector<GaussianMixture *> *ptr, UseType u = USE);
    void setpGaussPool(vector<Gaussian *> *ptr, UseType u = USE);
    /* Delete i-th state */
    void deleteState(int i);
    /* Delete state with state index = idx in USE*/
    bool deleteStateIdx(int idx);
    /* All v_state[:] with value > idx is decreased by 1 */
    bool cancelStateIdx(int idx);
    /* Set transition probability floor */
    static void setTRANS_ZERO(double z) { TRANS_ZERO = z; }
    static void setAllowedNDel(int n) { allowedNDel = n; }
    /* Insert a new state */
    int insertState(int state_idx);
    /****************************************************
     * CopyForThread(...) will copy parameters:
     * i_nstate, use, v_state, left, right, 
     * pi, v_trans, accum_ij, guass_isUsed, state_isUsed
     * !! pStatePool, pGaussPool, is the same as model's
     ****************************************************/
    void CopyForThread(const HMM_GMM &model);
    /********** Clear value **************/
    void ClearPi(UseType u, double val);
    void ClearIJ(double val);
    void ClearTrans(double val);
    /**************** EM *****************/
    void SyncUsed();
    void EMInitIter();
    /* FIXME: elaborate on the difference of following 3 */
    template<typename _Tp>
      double EMObs(typename vector<vector<_Tp> >::const_iterator start,
                   typename vector<vector<_Tp> >::const_iterator end,
                   int dim, double weight = 1.0, UpdateType udtype = UpdateAll);
    template<typename _Tp>
      double EMObsBound(typename vector<vector<_Tp> >::const_iterator start,
                        typename vector<vector<_Tp> >::const_iterator end,
                        int dim, Labfile *p_reflabfile, double weight = 1.0,
                        UpdateType udtype = UpdateAll );

    double EMObs(float **obs, int nframe, int dim, double weight = 1.0, UpdateType udtype = UpdateAll );
    double EMObsLabel(float **obs, int nframe, int dim, vector<int> *p_state_seq = NULL, double weight = 1.0, UpdateType udtype = UpdateAll );
    double EMObsBound(float **obs, int nframe, int dim, Labfile *p_reflabfile, double weight = 1.0, UpdateType udtype = UpdateAll );
    void AccumFromThread(const HMM_GMM &model);
    void EMUpdate(set<int> *p_delete_list = NULL, double backoff_weight = 0.0, UpdateType udtype = UpdateAll);
    double normPi(UseType u = USE);
    void normTrans();
    void normTransOther();
    void normTransOther(int sno);
    void normRTrans();
    double normOccupation();
    void dump_var();
    double getBjOt(unsigned int state_index, unsigned int obs_index) const {
      assert(state_index < bjOt.size());
      assert(obs_index < bjOt[state_index].size());
      return bjOt[state_index][obs_index];
    }
    /************** Viterbi **************/
    template<typename _Tp>
      void CalLogBgOt(_Tp** obs, int nframe, int dim);

    template<typename _Tp>
      void CalLogBgOt(typename vector<vector<_Tp> >::const_iterator start,
                      typename vector<vector<_Tp> >::const_iterator end,
                      int dim);

    void CalLogBjOtPxs(int nframe);
    void CalLogBjOt(int nframe);
    template<typename _Tp> void CalLogBjOt(int nframe, TwoDimArray<_Tp> *table);
    void CalLogAlpha(int nframe, vector<int> *p_state_seq = NULL);
    void ViteInit();
    double CalDelta(vector<int> &, bool isEnd);
    double CalLogDelta(vector<int>& state_seq,
                       vector<float>* likelihood_seq,
                       const vector<int> *p_endf); // Use pi, bjOt, v_trans
    double CalLogDelta(vector<int>& state_seq,
                       vector<float>* likelihood_seq = NULL); // Use pi, bjOt, v_trans
    void CalLogPrO(int nframe, vector<int> *p_label = NULL);

    /************** Some special functions **********/
    void CalLogCondToPostBjOt();
    void CalLogCondToLogPostBjOt();
    /************** I/O **************/
    void SaveHMM(FILE *fp, const DataType type);
    void LoadHMM(FILE* fp);
    void LoadHMM(ifstream& ifs);

  private:
    void Init();
    void ReadAscii(FILE *fp);
    void ReadAscii(ifstream& ifs);
    void SyncLeft();

    /* Log scale calculation */
    void CalLogAlphaBound(int nframe, vector<int> *p_endf);
    void CalLogBeta(int nframe, vector<int> *p_state_seq = NULL);
    void CalLogBetaBound(int nframe, vector<int> *p_startf);
    void CalLogGamma(int nframe);
    void CalLogEpsilon(int nframe, vector<int> *p_label = NULL);
    void CalLogEpsilonBound(int nframe, vector<int> *p_endf);
    void LogPi();
    void LogTrans();
    void ExpPxs();
    void ExpGamma();
    void ExpEpsilon();

    void AccumPi(vector<int> *p_state_seq = NULL, double obs_weight = 1.0);
    void AccumIJ(int nframe, vector<int> *p_state_seq = NULL,
                 double obs_weight = 1.0);
    void AccumWeightGaussian(float **obs, int nframe, int dim,
                             UpdateType udtype, vector<int> *p_label = NULL,
                             double obs_weight = 1.0);
    void accum2Trans();

    template<typename _Tp>
      void AccumWeightGaussian(
          typename vector<vector<_Tp> >::const_iterator start,
          typename vector<vector<_Tp> >::const_iterator end,
          int dim,
          UpdateType udtype,
          vector<int> *p_label = NULL,
          double obs_weight = 1.0);

    // EM memory
    // Index: t (time), i (state), j (state), x (mixture)
    vector<vector<double> > bgOt;    // Pr(Ot|Gaussian = g), (g,t)
    vector<vector<double> > bjOt;    // Pr(Ot|State_t = i),  (i,t)
    vector<vector<vector<double> > > px_s; // Pr(mix = x | State_t = i, O), (i,x,t)
    vector<vector<double> > alpha;   // Pr(O{1~t},State_t=i),   (i,t)
    vector<vector<double> > beta;    // Pr(O{t+1~T}|State_t=i), (i,t)
    vector<vector<double> > delta;   // Max Pr(O{1~t},State_t=i),   (i,t)
    vector<vector<double> > gamma;   // Pr(State_t = i|O),      (i,t)
    vector<vector<vector<double> > > epsilon; // Pr(State_t=i,State_t+1=j|O), (i,j,t)
    vector<vector<double> > accum_ij;
    vector<double> occupation;
    vector<bool> gauss_isUsed;
    vector<bool> state_isUsed;
    int nframe;
    double prO;

    // Data
    int i_nstate;
    int use;
    vector<int> v_state;                      // vector of state index
    vector<int> left, right;
    vector<vector<int> > v_left;
    vector<double> pi[2];                     // initial prob
    vector<vector<double> > v_trans;          // transition prob
    vector<vector<double> > v_rtrans;         // reverse trans prob
    string s_name;
    bool isLog;
    pthread_mutex_t H_mutex;

    vector<GaussianMixture *> *pStatePool[2]; // pointer to state pool
    vector<Gaussian *> *pGaussPool[2];        // pointer to state pool
    double pdf_weight;
    static double TRANS_ZERO;
    static int allowedNDel;
};/*}}}*/

template<typename _Tp>
void HMM_GMM::CalLogBgOt(_Tp **obs, int nframe, int dim) {/*{{{*/
  vector<Gaussian *> &vGauss = *(getpGM(0,USE)->getpGaussPool());

  bgOt.resize(vGauss.size());
  for (unsigned g = 0; g < vGauss.size(); g++) {
    if (!gauss_isUsed[g]) {
      bgOt[g].clear();
      continue;
    }
    bgOt[g].resize(nframe);
    for (int t = 0; t < nframe; t++)
      bgOt[g][t] = vGauss[g]->logProb(obs[t],dim,true) * pdf_weight;
    //bgOt[g][t] = vGauss[g]->logProb(obs[t],dim,true);
  }
}/*}}}*/

template<typename _Tp>
void HMM_GMM::CalLogBgOt(typename vector<vector<_Tp> >::const_iterator start, /*{{{*/
                         typename vector<vector<_Tp> >::const_iterator end,
                         int dim) {
  vector<Gaussian *> &vGauss = *(getpGM(0,USE)->getpGaussPool());

  int nframe = end - start;
  bgOt.resize(vGauss.size());
  for (unsigned g = 0; g < vGauss.size(); g++) {
    if (!gauss_isUsed[g]) {
      bgOt[g].clear();
      continue;
    }
    bgOt[g].resize(nframe);
    for (int t = 0; t < nframe; t++) {
      typename vector<vector<_Tp> >::const_iterator itr = start + t;
      const _Tp* obs = &itr->front();
      bgOt[g][t] = vGauss[g]->logProb(obs, dim, true) * pdf_weight;
    }
  }
}/*}}}*/

template<typename _Tp>
double HMM_GMM::EMObs(typename vector<vector<_Tp> >::const_iterator start,
                      typename vector<vector<_Tp> >::const_iterator end,
                      int dim, double weight, UpdateType udtype) {
  int nframe = end - start;
  CalLogBgOt<_Tp>(start, end, dim); // Use gauss_isUsed; +bgOt
  CalLogBjOtPxs(nframe);            // Use bgOt; +bjOt, +px_s
  CalLogAlpha(nframe);              // Use trans pi bjot; +alpha
  CalLogBeta(nframe);               // Use trans bjot; +beta
  CalLogPrO(nframe);                // Use alpha; +prO;
  CalLogGamma(nframe);              // Use alpha beta prO; +gamma
  CalLogEpsilon(nframe);            // Use alpha beta trans bjot prO; +epsilon = logp(O, qt:i->j)
  ExpPxs();
  ExpGamma();
  ExpEpsilon();
  AccumPi(NULL, weight);            // Use gamma; +pi
  AccumIJ(nframe, NULL, weight);    // Use epsilon; +acuum_ij
  AccumWeightGaussian<_Tp>(start, end, dim, udtype, NULL, weight);
  // Use gamma px_s; (+e)state->setWeight(), (+e)gauss->AddData()

  return weight * prO;
}

template<typename _Tp>
double HMM_GMM::EMObsBound(typename vector<vector<_Tp> >::const_iterator start,
                           typename vector<vector<_Tp> >::const_iterator end,
                           int dim, Labfile *p_reflabfile, double weight,
                           UpdateType udtype) {

  int nframe = end - start;
  CalLogBgOt<_Tp>(start, end, dim); // Use gauss_isUsed; +bgOt
  CalLogBjOtPxs(nframe);            // Use bgOt; +bjOt, +px_s
  CalLogAlphaBound(nframe, p_reflabfile->getpEndf());
  // Use trans pi bjot; +alpha
  CalLogBetaBound(nframe, p_reflabfile->getpStartf()); // Use trans bjot; +beta
  CalLogPrO(nframe);                // Use alpha; +prO;
  CalLogGamma(nframe);              // Use alpha beta prO; +gamma
  CalLogEpsilonBound(nframe ,p_reflabfile->getpEndf()); 
  // Use alpha beta trans bjot prO; +epsilon
  ExpPxs();
  ExpGamma();
  ExpEpsilon();
  AccumPi(NULL, weight);            // Use gamma; +pi
  AccumIJ(nframe, NULL, weight);    // Use epsilon; +acuum_ij
  AccumWeightGaussian<_Tp>(start, end, dim, udtype, NULL, weight);
  // Use gamma px_s; (+e)state->setWeight(), (+e)gauss->AddData()

  return weight * prO;
}/*}}}*/

template<typename _Tp>
void HMM_GMM::AccumWeightGaussian(
    typename vector<vector<_Tp> >::const_iterator start,
    typename vector<vector<_Tp> >::const_iterator end,
    int dim,
    UpdateType udtype,
    vector<int> *p_label,
    double obs_weight) {
  bool useLabel = (p_label != NULL);

  int nframe = end - start;

  /* For each state */
  for (unsigned i = 0; i < gamma.size(); i++) {

    int sno_i = useLabel ? (*p_label)[i] : i;
    GaussianMixture *state = getpGM(sno_i,UNUSE);

    /* For each gaussian associated with this state*/
    for (int x = 0; x < state->getNmix(); x++) {
      Gaussian *pg = state->getpGauss(x);
      for (int t = 0; t < nframe; t++) {
        typename vector<vector<_Tp> >::const_iterator itr = start + t;
        const _Tp* obs = &(*itr)[0];
        double gamma_i_x_t = obs_weight * gamma[sno_i][t] * px_s[v_state[sno_i]][x][t];
        if (gamma_i_x_t <= ZERO) continue;
        pg->AddData(obs, dim, gamma_i_x_t, udtype);
        // Here we do not divide by sum_gamma[i] because
        // the total sum of weights in state i is
        // actually sum_gamma[i]
        state->setWeight(x, gamma_i_x_t, ADD);
      }
    }
  }

}



void SaveHMMGMG(string filename, HMM_GMM &model);
void LoadHMMGMG(string filename, HMM_GMM *model, vector<GaussianMixture*> p_statePool[2], vector<Gaussian*> p_gaussPool[2]);
void LoadHMMGMG(string filename, HMM_GMM *model, vector<GaussianMixture*> &statePool, vector<Gaussian*> &gaussPool);
bool DeleteState(unsigned idx, HMM_GMM &model, set<int> &recycle_state, set<int> &recycle_gauss );
void RemoveTrash(HMM_GMM &model, vector<GaussianMixture*> *statePool, vector<Gaussian*> *gaussPool, set<int> &recycle_state, set<int> &recycle_gauss);
/************************************************
 * NewStateCopy(...) will call GetState() to 
 * append at the end of pStatePool (might also 
 * append at the end of pGaussPool) new USE|UNUSE
 * states and do a copy on USE only:
 * model.getpStatePool(USE)[sid]
 *          => model.getpStatePool(UNUSE)[nsid]
 * (copy Gaussian mixture, too)
 ************************************************/
int NewStateCopy(int sid, HMM_GMM &model, set<int> *p_state_recycler, set<int> *p_gauss_recycler);
/************************************************
 * GetState(...) append new USE|UNUSE states in 
 * model.getpStatePool(USE|UNUSE)[nsid] with 
 * specified `dim', call nummix times of GetGaussian(),
 * and return `nsid'.
 ************************************************/
int GetState(HMM_GMM &model, set<int> *p_state_recycler, set<int> *p_gauss_recycler, const int dim, const int num_mix );
int GetGaussian(HMM_GMM &model, set<int> *p_gauss_recycler, int dim );

#endif
