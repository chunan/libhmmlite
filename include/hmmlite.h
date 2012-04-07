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
#include "ugoc_utility.h"
#include "libatlas_wrapper.h"

#define float_inf std::numeric_limits<float>::infinity()


const double LOG2PI  = 1.83787707;
const double PI      = 3.14159265358979;
const double TPI     = 6.28318530717959;     /* PI*2 */
const double LZERO   = -1.0E10;    /* ~log(0) */
const double LSMALL  = -0.99E10;
const double MINEARG = -708.3;     /* lowest exp() arg  = log(MINLARG) */
const double MINLARG = 2.45E-308;  /* lowest log() arg  = exp(MINEARG) */
const double ZERO    = 1e-20;
enum UseType { USE, UNUSE};
enum SetType { ADD, SET};
enum DataType{ ASCII, BINARY};
enum UpdateType { UpdateAll, UpdateMean, UpdateCov };

using namespace std;
using namespace atlas;

double LAdd(double x, double y);
double LSub(double x, double y);
double LProd( double loga, double logb);
double LDiv( double loga, double logb);
double LOG( double a);
double EXP( double a);

inline bool isEqual( double a, double b ){ return fabs(a - b) <= ( (a < b) ? fabs(a) : fabs(b) ) * std::numeric_limits<double>::epsilon(); }

class Labfile { /*{{{*/
  public:
    Labfile(){ Init(); }
    Labfile(string filename){ LoadFile(filename); }
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
    void DumpData(){/*{{{*/
      cout << "====== Labfile ======\n";
      for(int i = 0; i < num_lab; i++) {
        cout << "S" << i << "\ts" << start_f[i] << "\te" << end_f[i] << '\t' << cluster[i];
        if (!score.empty()) cout << score[i];
        cout << endl;
      }
      cout << "num_lab: " << num_lab << endl;
    }/*}}}*/
    void DumpData(vector<int> &seg_head_state, vector<int> &seg_tail_state){/*{{{*/
      assert(seg_head_state.size() == seg_tail_state.size());
      cout << "====== Labfile ======\n";
      for(unsigned s = 0; s < seg_head_state.size(); s++){
        for(int i = seg_head_state[s]; i <= seg_tail_state[s]; i++){
          cout << "S" << i << "\ts" << start_f[i] << "\te" << end_f[i] << '\t' << cluster[i] << "\tSeg" << s << endl;
        }
      }
      cout << "num_lab: " << num_lab << endl;
    }/*}}}*/
    void parseStateSeq( vector<int> &state_seq );
    void parseStateSeq( vector<int> &state_seq, vector<int> &ref_end_f );
    void expandMaxClust(unsigned *p_max_clust){/*{{{*/
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
    int getCluster(int t) const { return cluster[t]; }
    float getScore(int t) const { return score[t]; }
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

class Gaussian /*{{{*/
{
  public:
    Gaussian(){ Init(); }
    Gaussian(const Gaussian &g);
    Gaussian(int d){ Init(); AllocateMem(d); }
    ~Gaussian(){ DeallocateMem(); }
    void AllocateMem(int d = 0);
    double InvertCov();
    double getLogConst(){ return logConst; }
    void ZeroCov(){p_cov->zeroFill();}
    void ZeroMean(){p_mean->zeroFill();}
    void AddData(const double *data, const int dim, const double prob, UpdateType udtype = UpdateAll);
    void AddMeanCov(const Gaussian &g);
    const Gaussian & operator=(const Gaussian &g);
    double Bhat_dist(const Gaussian &g1, const Gaussian &g2);
    void CopyCov(const Gaussian &g);
    void CopyMean(const Gaussian &g);
    void backoff(const Gaussian &g, const double backoff_weight);
    bool normMeanCov(const bool subtractmean, UpdateType udtype, double N = -1);
    void setMean(const int idx, const double val){ p_mean->setEntry(idx,0,val); }
    void setCov(const int r, const int c, const double val){ assert(r <= c); p_cov->setEntry(r,c,val); }
    void setDiag(const bool d){ isDiag = d; }
    /* Set REQUIRED_FRAME for normMeanCov() */
    static void setRequiredFrame(double r){ REQUIRED_FRAME = r; }
    static void setVarFloor(double v){ VAR_FLOOR = v; }
    static double getRequiredFrame(){ return REQUIRED_FRAME; }
    static double getVarFloor(){ return VAR_FLOOR; }

    int getDim() const { return dim; }
    double getMean(const int idx) const { return p_mean->entry(idx,0); }
    double getCov(int r, int c) const { if(r > c)swap(r,c); return  p_cov->entry(r,c); }
    double getWeight() const { return weight; }
    bool getDiag() const { return isDiag; }
    double getTotalVar() const;
    void ClearGaussian();
    void AddVarFloor();
    template<typename _Tp>
    double logProb(_Tp *data, const int dim, const bool islog = true) const;
    template<typename _Tp1, typename _Tp2>
    double Bhat(_Tp1 *data1, const _Tp2 *data2, const int dim) const;
    void display(FILE *fp) const;
    void SaveGaussian(FILE *fp, const DataType type);
    void LoadGaussian(FILE *fp);

  private:
    void Init();
    void DeallocateMem(){
      if(p_mean) delete p_mean;
      if(p_cov)  delete p_cov;
      if(p_icov) delete p_icov;
      Init();
    }
    void ReadAscii(FILE *fp);
    void ReadBinary(FILE *fp);

    // Data
    int dim;
    Matrix *p_mean;
    Matrix *p_cov;
    Matrix *p_icov;
    int numframe;
    double weight;
    double logConst; // -0.5 * log(det(Covariance))
    bool isDiag;
    pthread_mutex_t G_mutex;

    static double REQUIRED_FRAME;
    static double VAR_FLOOR;
};
/*}}}*/

class GaussianMixture /*{{{*/
{
  public:
    GaussianMixture(){Init(0,0,0);}
    GaussianMixture(int d, int m, vector<Gaussian *> *p) {Init(d,m,p);}
    ~GaussianMixture(){}
    void Init(int d, int x, vector<Gaussian *> *p);
    const GaussianMixture & operator=(const GaussianMixture & gm);
    // Get value
    int getDim() const { return dim; }
    int getGaussIdx(unsigned x) const { assert(x < v_gaussidx.size()); return v_gaussidx[x]; }
    double getWeight(unsigned x) const { assert(x < v_gaussidx.size()); return v_weight[x]; }
    Gaussian *getpGauss(unsigned x) const { assert(x < v_gaussidx.size()); return (*pGaussPool)[getGaussIdx(x)]; }
    vector<Gaussian *> *getpGaussPool() const { return pGaussPool; }
    int getNmix() const { return v_gaussidx.size(); }
    string getName() const { return s_name;}
    bool containGidx(int gidx);
    static double getRequiredFrame(){ return REQUIRED_FRAME; }
    // Set value
    void setDim(int d){ dim = d; }
    void setNmix(int x) { v_weight.resize(x); v_gaussidx.resize(x); }
    void setGaussIdx(unsigned x, unsigned idx);
    void copyGaussIdx(const GaussianMixture *pstate);
    void setpGaussPool(vector<Gaussian *> *ptr){ pGaussPool = ptr;}
    void setWeight(int x, double val, SetType s); 
    void copyWeight(GaussianMixture &s); 
    void UniformWeight();
    bool cancelGaussIdx(int idx);
    void setName(const char *str){ s_name = str; }
    void setName(const string str){ s_name = str; }
    static void setRequiredFrame(double r){ REQUIRED_FRAME = r; }
    // Clear value
    void ClearWeight();
    // Normalization
    bool normWeight(double &weightsum);
    void display(FILE *fp) const;
    void SaveGaussianMixture(FILE *fp, const DataType type = ASCII);
    void LoadGaussianMixture(FILE *fp);
  private:
    void ReadAscii(FILE *fp);
    void ReadBinary(FILE *fp);

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
    HMM_GMM(){Init();}
    ~HMM_GMM(){};
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
    double getPrO() const {return prO; }
    /* Dump all EM-related variables */
    void dump_param() const;
    void display(FILE *fp) const;

    /************ Set value **************/
    /* Set number of state */
    void setNstate(int n);
    /* Set use */
    void setUse(int u){ assert(u==0 || u==1); use = u;}
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
    void ClearPi(UseType u);
    void ClearIJ();
    void ClearTrans();
    /**************** EM *****************/
    void SyncUsed();
    void EMInitIter();
    /* FIXME: elaborate on the difference of following 3 */
    double EMObs(double **obs, int nframe, int dim, double weight = 1.0, UpdateType udtype = UpdateAll );
    double EMObsLabel(double **obs, int nframe, int dim, vector<int> *p_state_seq = NULL, double weight = 1.0, UpdateType udtype = UpdateAll );
    double EMObsBound(double **obs, int nframe, int dim, Labfile *p_reflabfile, double weight = 1.0, UpdateType udtype = UpdateAll );
    void AccumFromThread(const HMM_GMM &model);
    void EMUpdate(set<int> *p_delete_list = NULL, double backoff_weight = 0.0, UpdateType udtype = UpdateAll);
    double normPi(UseType u = USE);
    void normTrans();
    void normTransOther();
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
    void CalLogBjOtPxs(int nframe);
    void CalLogAlpha(int nframe, vector<int> *p_state_seq = NULL);
    void ViteInit();
    double CalDelta(vector<int> &, bool isEnd);
    double CalLogDelta(vector<int> &state_seq, const vector<int> *p_endf = NULL); // Use pi, bjOt, v_trans
    void CalLogPrO(int nframe, vector<int> *p_label = NULL);
    template<typename _Tp>
    void CalLogBjOt(int nframe, TwoDimArray<_Tp> *table);

    /************** Some special functions **********/
    void CalLogCondToPostBjOt();
    void CalLogCondToLogPostBjOt();
    /************** I/O **************/
    void SaveHMM(FILE *fp, const DataType type);
    void LoadHMM(FILE *fp);

  private:
    void Init();
    void ReadAscii(FILE *fp);
    void ReadBinary(FILE *fp);
#ifdef NODEF
    void CalBgOt(double **obs, int nframe, int dim);
    void CalBjOtPxs();
    void CalAlpha();
    void CalBeta();
    void CalPrO();
    void CalGamma();
    void CalSumGamma();
    void CalEpsilon();
#endif

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

    void AccumPi( vector<int> *p_state_seq = NULL, double obs_weight = 1.0);
    void AccumIJ( int nframe, vector<int> *p_state_seq = NULL, double obs_weight = 1.0);
    void AccumWeightGaussian(double **obs, int nframe, int dim, UpdateType udtype, vector<int> *p_state_seq = NULL, double obs_weight = 1.0);
    void accum2Trans();

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
void HMM_GMM::CalLogBgOt(_Tp **obs, int nframe, int dim)/*{{{*/
{
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
double Gaussian::logProb(_Tp* data, const int dim, const bool islog) const/*{{{*/
{
  double logpr = logConst - 0.5 * Bhat(data, p_mean->pointer(), dim);

  if (islog) return logpr;
  else return exp(logpr);

}/*}}}*/

template<typename _Tp1, typename _Tp2>
double Gaussian::Bhat(_Tp1 *data1, const _Tp2 *data2, const int dim) const/*{{{*/
{
  /* Memory arrangement */
  double *data_hat = new double[dim];
  /**********************/

  double xAx = 0.0;
  for (int r = 0; r < dim; r++) data_hat[r] = data1[r] - data2[r];
  for (int r = 0; r < dim; r++) {
    xAx += data_hat[r] * data_hat[r] * p_icov->entry(r,r);
    for (int c = r+1; c < dim; c++) {
      xAx += 2.0 * data_hat[r] * data_hat[c] * p_icov->entry(r,c);
    }
  }
  delete [] data_hat;
  return xAx;

}/*}}}*/

template<typename _Tp>
void HMM_GMM::CalLogBjOt(int nframe, TwoDimArray<_Tp> *ptab) {/*{{{*/
  TwoDimArray<_Tp>& table = *ptab;
  table.Resize(i_nstate, nframe);
  table.Memfill(LZERO);
  /* For each state */
  for (int j = 0; j < i_nstate; ++j) {
    GaussianMixture* state = getpGM(j,USE);
    int mixsize = state->getNmix();
    for (int t = 0; t < nframe; ++t) {
      /* sum_{x = 1}^{mixsize} (weight_x * bgot) */
      for (int x = 0; x < mixsize; ++x) {
        _Tp bgot =  LProd(LOG(state->getWeight(x)),
                          bgOt[state->getGaussIdx(x)][t]);
        table(j, t) = LAdd(table(j, t), bgot);
      }
    }
  }
}/*}}}*/

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
