
double Gaussian::REQUIRED_FRAME = 30;
double Gaussian::VAR_FLOOR = 0.0;

Gaussian::Gaussian(const Gaussian &g) {/*{{{*/
  Init();
  AllocateMem(g.getDim());
  *p_mean = *g.p_mean;
  *p_cov  = *g.p_cov;
  *p_icov = *g.p_icov;
  logConst = g.logConst;
}/*}}}*/

void Gaussian::Init() {/*{{{*/
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

  double logdet = 0.0;

  if (isDiag) { /* Diag covariance */
    p_icov->zeroFill();
    for (int r = 0; r < dim; ++r) {
      double num = p_cov->entry(r, r);
      if (num > 0) {
        logdet += log(num);
        p_icov->setEntry(r, r, 1 / num);
      } else {
        logdet = std::numeric_limits<double>::infinity();
        break;
      }
    }
    logConst = -logdet / 2;

  } else { /* Full covariance */

    *p_icov = *p_cov;
    logdet = AisCholeskySymmetricA(p_icov);
    if (logdet != -std::numeric_limits<double>::infinity()) {
      AisInvCholeskyA(p_icov);
      logConst = - logdet / 2;
    }
    else {
      logdet = std::numeric_limits<double>::infinity();
    }

  }

  pthread_mutex_unlock(&G_mutex);

  return logdet;
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

  for (int i = 0; i < dim; i++) {
#if 0
    if (p_cov->entry(i, i) < VAR_FLOOR) {
      p_cov->setEntry(i,i,VAR_FLOOR);
    }
#endif
    p_cov->setEntryPlus(i, i, VAR_FLOOR);
  }

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

double Bhat_dist(const Gaussian &g1, const Gaussian &g2) {/*{{{*/
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


    void setMean(const int idx, const double val) {
      p_mean->setEntry(idx,0,val);
    }

    void setCov(const int r, const int c, const double val) {
      assert(r <= c);
      p_cov->setEntry(r,c,val);
    }

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


  if (isDiag) {
    fprintf(fp,"diag\n");
  } else {
    fprintf(fp,"full\n");
  }

  fprintf(fp,"cov:\n");
  for (int r = 0; r < dim; r++) {
    fprintf(fp," %g",p_cov->entry(r,r));
    if (!isDiag) {
      for (int c = r + 1; c < dim; c++) {
        fprintf(fp," %g",p_cov->entry(r,c));
      }
    }
    fprintf(fp,"\n");
  }


  if (fp == stdout || fp == stderr) {
    cout << "icov:\n";
    for (int r = 0; r < dim; r++) {
      cout << fixed << p_icov->entry(r,r) << ' ';
      if (!isDiag) {
        for (int c = r + 1; c < dim; c++)
          cout << fixed << p_icov->entry(r,c) << ' ';
      }
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

istream& operator<<(istream& is) {/*{{{*/

  string line, tag;
  Getline(is, line);
  stringstream iss(line);

  iss >> tag;
  assert(tag.compare("Gaussian") == 0);

  iss >> tag;
  if (tag.compare("ascii") == 0) {
    ReadAscii(is);
  } else {
    fprintf(stderr,"Unknown tag for Gaussian, only ascii/binary allowed\n");
  }
}/*}}}*/

void Gaussian::ReadAscii(ifstream& ifs) {/*{{{*/

  string line, tag;
  int i_val;
  float f_val;

  while (Getline(ifs, line)) {
    stringstream iss(line);
    iss >> tag;
    if (tag.compare("EndGaussian") == 0) {
      break;

    } else if (tag.compare("name:") == 0) {
      iss >> name;

    } else if (tag.compare("dim:") == 0) {
      iss >> i_val;
      AllocateMem(i_val);

    } else if (tag.compare("mean:") == 0) {
      for (int r = 0; r < dim; r++) {
        iss >> f_val;
        p_mean->setEntry(r, 0, static_cast<double>(f_val));
      }

    } else if (tag.compare("cov") == 0) {

      iss >> tag;
      if (tag.compare("diag:") == 0)
        isDiag = true;
      else if (tag.compare("full:") == 0)
        isDiag = false;
      else
        cerr << "Unknown flag \"mean " << tag << "\"\n";

      for (int r = 0; r < dim; r++) {
        Getline(ifs, line);
        iss.str(line);
        iss.clear();
        iss >> f_val;
        p_cov->setEntry(r, r, static_cast<double>(f_val));
        if (!isDiag) {
          for (int c = r + 1; c < dim; c++) {
            iss >> f_val;
            p_cov->setEntry(r,c,static_cast<double>(f_val));
          }
        }
      }

    } else {
      ErrorExit(__FILE__, __LINE__, 1, "unknown tag ``%s''\n", tag.c_str());
    }
  }
}/*}}}*/

