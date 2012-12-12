#include "hmmlite.h"
#include <unistd.h>

int main(int argc, const char *argv[]) {
  HMM_GMM model;
  vector<Gaussian *> gaussPool[2];
  vector<GaussianMixture *> statePool[2];

  LoadHMMGMG("model.hmmlite", &model, statePool, gaussPool);
  model.SyncUsed();
  unsigned ngm, ng;
  ngm = ng = 0;
  for (unsigned g = 0; g < gaussPool[0].size(); g++) {
    if (model.getGisUsed(g)) ng++;
  }
  cout << "ng = " << ng << "pool = " << gaussPool[2].size() << endl;
  for (unsigned s = 0; s < statePool[0].size(); s++) {
    if (model.getSisUsed(s)) ngm++;
  }
  cout << "ngm = " << ngm << "pool = " << statePool[2].size() << endl;


  return 0;
}
