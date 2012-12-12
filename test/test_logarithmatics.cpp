#include <iostream>
#include "../include/logarithmatics.h"

using namespace std;

int main(int argc, const char *argv[]) {

  LLDouble z;
  cout << "z = " << z << endl;

  /* Initialize lin */
  LLDouble a(19, LLDouble::LINDOMAIN);
  cout << "a = " << a << endl;
  /* lin -> log */
  a.to_logdomain();
  cout << "a = " << a << endl;

  LLDouble b(100.0, LLDouble::LINDOMAIN);
  b.to_logdomain();
  cout << "b = " << b << endl;

  cout << "a += b " << (a += b) << endl;
  cout << "a = " << a.to_lindomain() << endl;


  return 0;
}
