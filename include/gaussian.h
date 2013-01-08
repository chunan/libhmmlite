#ifndef __GAUSSIAN_H__
#define __GAUSSIAN_H__

double Bhat_dist(const Gaussian &g1, const Gaussian &g2);

class Gaussian { /*{{{*/
  public:
    enum CovType {DIAG, FULL};
    Gaussian(const Gaussian &g);
    Gaussian(int d, CovType type);
    ~Gaussian() {}

    // I/O
    friend istream& operator>>(istream& is, const Gaussian& g);
    friend ostream& operator>>(ostream& os, const Gaussian& g);

    // mutators
    void ZeroCov();
    void ZeroMean();
    void ZeroGaussian();
    void setMean(const int idx, const double val);
    void setCov(const int r, const int c, const double val);
    const Gaussian & operator=(const Gaussian &g);
    void CopyCov(const Gaussian &g);
    void CopyMean(const Gaussian &g);
    void backoff(const Gaussian &g, const double backoff_weight);

    // accessors
    double getLogConst() { return logConst; }

    int getDim() const { return dim; }
    double getMean(const int idx) const { return p_mean->entry(idx,0); }
    double getCov(int r, int c) const { if(r > c)swap(r,c); return  p_cov->entry(r,c); }
    bool getDiag() const { return isDiag; }
    double getTotalVar() const;
    void AddVarFloor();
    template<typename _Tp>
      double logProb(const _Tp *data, int dim, bool islog = true) const;
    template<typename _Tp1, typename _Tp2>
      double Bhat(const _Tp1 *data1, const _Tp2 *data2, const int dim) const;
    void open(string filename);
    void save(string filename);
    void LoadGaussian(ifstream& ifs);
    void SaveGaussian(FILE* fp, const DataType type);

  private:
    void Init();
    double InvertCov();
    void ReadAscii(ifstream& ifs);

    // Data
    string name_;
    int dim_;
    CovType ctype_;
    Matrix mean_;
    Matrix cov_;
    Matrix icov_;
    double logConst_; // -0.5 * log(det(Covariance))
    bool icov_ready_;
    bool protect_muti_acc_;
    // multi-thread protection
    pthread_mutex_t G_mutex;
};
/*}}}*/

#include <gaussian.hpp>

#endif /* __GAUSSIAN_H__ */
