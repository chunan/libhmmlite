
template<typename _Tp>
double Gaussian::logProb(const _Tp* data, int dim, bool islog) const/*{{{*/
{
  double logpr = logConst - 0.5 * Bhat(data, p_mean->pointer(), dim);

  if (islog) return logpr;
  else return exp(logpr);

}/*}}}*/

template<typename _Tp1, typename _Tp2>
double Gaussian::Bhat(const _Tp1 *data1, const _Tp2 *data2, const int dim) const/*{{{*/
{
  /* Memory arrangement */
  vector<double>data_hat(dim);
  /**********************/

  double xAx = 0.0;
  for (int r = 0; r < dim; r++) data_hat[r] = data1[r] - data2[r];
  for (int r = 0; r < dim; r++) {
    xAx += data_hat[r] * data_hat[r] * p_icov->entry(r,r);
    if (!isDiag) {
      for (int c = r+1; c < dim; c++) {
        xAx += 2.0 * data_hat[r] * data_hat[c] * p_icov->entry(r,c);
      }
    }
  }
  return xAx;

}/*}}}*/

template<typename _Tp>
void Gaussian::AddData(_Tp *data, /*{{{*/
                       const int dim,
                       const double prob,
                       UpdateType udtype) {
  if (!(prob >= 0)) {
    ErrorExit(__FILE__,__LINE__,-1,"Illigal prob (%f) in Gaussian::AddData()\n",prob);
  }

  if (prob < ZERO) return;

  pthread_mutex_lock(&G_mutex);

  numframe++;
  weight += prob;
  if (udtype == UpdateAll) {
    for (int r = 0; r < dim; r++) {
      double val = prob * data[r];
      p_mean->setEntryPlus(r, 0, val);
      p_cov->setEntryPlus(r,r, val * data[r]);
      if (!isDiag) {
        for (int c = r+1; c < dim; c++)
          p_cov->setEntryPlus(r,c, val * data[c]);
      }
    }
  } else if (udtype == UpdateMean) {
    for (int r = 0; r < dim; r++) {
      p_mean->setEntryPlus(r, 0, prob * data[r]);
    }
  } else if (udtype == UpdateCov) {
    assert(false);
  }

  pthread_mutex_unlock(&G_mutex);

}/*}}}*/

template<typename _Tp>
void HMM_GMM::CalLogBjOt(int nframe, TwoDimArray<_Tp> *ptab) {/*{{{*/
  TwoDimArray<_Tp>& table = *ptab;
  vector<GaussianMixture*>& gmpool = *pStatePool[use];
  table.Resize(gmpool.size(), nframe);
  table.Memfill(LZERO);

  /* For each state */
  for (unsigned j = 0; j < gmpool.size(); ++j) {
    if (!state_isUsed[j]) continue;
    int mixsize = gmpool[j]->getNmix();

    if (mixsize == 1) { // single-Gaussian
      for (int t = 0; t < nframe; ++t)
        table(j, t) = bgOt[gmpool[j]->getGaussIdx(0)][t];

    } else { // multi-Gaussian
      vector<double> weight(gmpool[j]->getWeight());
      for_each(weight.begin(), weight.end(), LOG);
      for (int t = 0; t < nframe; ++t) { /* for each frame */
        for (int x = 0; x < mixsize; ++x) { /* for each Gaussian */
          int g = gmpool[j]->getGaussIdx(x);
          table(j, t) = LAdd(table(j, t), LProd(weight[x], bgOt[g][t]));
        } /* for each Gaussian */
      } /* for each frame */

    } /* if mixsize == 1, else */
  } /* for j */

  /* Recycle bgOt */
  bgOt.clear();
}/*}}}*/

