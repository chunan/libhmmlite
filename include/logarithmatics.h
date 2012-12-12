#ifndef __LOGARITHMATHICS_H__
#define __LOGARITHMATHICS_H__

#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>
#include "def.h"


/*                LIN      |      LOG
 * ------------------------+-----------------------
 * 2.45e-308 = MINLINARG <-+-> MINLOGARG = -708.3
 *         0 =           <-+-> LSMALL    = -0.5e10
 *         0 =           <-+-> LZERO     = -1e10
 * ------------------------+-----------------------
 * Log domain operation range: LSMALL ~ inf
 * If Log -> Lin, operation range: MINLOGARG ~ inf
 */
class LLDouble {
  public:
    friend LLDouble operator+(const LLDouble a, const LLDouble b);
    friend LLDouble operator-(const LLDouble a, const LLDouble b);
    friend LLDouble operator*(const LLDouble a, const LLDouble b);
    friend LLDouble operator/(const LLDouble a, const LLDouble b);
    friend std::ostream& operator<<(std::ostream& os, const LLDouble& ref);
  public:
    enum Type {LOGDOMAIN, LINDOMAIN};
    LLDouble() : _val(LZERO), _type(LOGDOMAIN) {}
    LLDouble(const LLDouble& ref) { *this = ref; }
    LLDouble(const double d, const Type t);
    LLDouble& to_logdomain();
    LLDouble& to_lindomain();
    const LLDouble& operator=(const LLDouble& ref);
    LLDouble& operator+=(const LLDouble& ref);
    LLDouble& operator-=(const LLDouble& ref);
    LLDouble& operator*=(const LLDouble& ref);
    LLDouble& operator/=(const LLDouble& ref);

  public:
    // Default gives Log zero
    static LLDouble LogZero() { return LLDouble(); }

  private:
    static const double LZERO;    /* ~log(0) */
    static const double LSMALL;
    static const double MINLOGARG;  /* lowest exp() arg  = log(MINLARG) */
    static const double MINLINARG;  /* lowest log() arg  = exp(MINEARG) */

  private:
    double LOG(double a);
    double EXP(double a);

  private:
    double _val;
    Type _type;
};


inline bool isEqual(double a, double b) {
  bool isequal = fabs(a - b) <=
    std::min<double>(fabs(a), fabs(b)) * std::numeric_limits<double>::epsilon();
  return isequal;
}

#endif /* __LOGARITHMATHICS_H__ */
