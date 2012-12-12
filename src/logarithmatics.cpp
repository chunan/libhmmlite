#include "logarithmatics.h"
#include "utility.h"
#include <iostream>
#include <cassert>


/* min log domain operatable point */
const double LLDouble::LZERO   = -1.0E10;   /* ~log(0) */
const double LLDouble::LSMALL  = -0.5E10;
/* min log num when convert to linear (= log(MINLARG)) */
const double LLDouble::MINLOGARG = -708.3;
const double LLDouble::MINLINARG = 2.45E-308;


LLDouble operator+(const LLDouble a, const LLDouble b) {/*{{{*/
  assert(a._type == b._type);

  if (a._type == LLDouble::LINDOMAIN) {
    return LLDouble(a._val + b._val, LLDouble::LINDOMAIN);
  } else {
    double x = a._val, y = b._val, diff;
    if (x < y) std::swap(x, y); // make sure x > y
    diff = y - x; // diff < 0
    if (diff < LLDouble::MINLOGARG) {
      return a;
    } else {
      return LLDouble(x + log(1.0 + exp(diff)), LLDouble::LOGDOMAIN);
    }
  }

}/*}}}*/

LLDouble operator-(const LLDouble a, const LLDouble b) {/*{{{*/
  assert(a._type == b._type);

  if (a._val < b._val) {
    ErrorExit(__FILE__, __LINE__, -1,
              "%f - %f < 0.0\n", a._val, b._val);
  }

  if (a._type == LLDouble::LINDOMAIN) {
    return LLDouble(a._val - b._val, LLDouble::LINDOMAIN);
  } else {
    double diff = b._val - a._val; // diff < 0.0
    if (diff < LLDouble::MINLOGARG) {
      return a;
    } else {
      double z = a._val + log(1.0 - exp(diff));
      if (z < LLDouble::LSMALL) z = LLDouble::LZERO;
      return LLDouble(z, LLDouble::LOGDOMAIN);
    }
  }

}/*}}}*/

LLDouble operator*(const LLDouble a, const LLDouble b) {/*{{{*/
  assert(a._type == b._type);

  if (a._type == LLDouble::LINDOMAIN) {
    return LLDouble(a._val * b._val, LLDouble::LINDOMAIN);
  } else {
    double z = a._val + b._val;
    return (z <= LLDouble::LSMALL)
      ? LLDouble::LogZero()
      : LLDouble(z, LLDouble::LOGDOMAIN);
  }
}/*}}}*/

LLDouble operator/(const LLDouble a, const LLDouble b) {/*{{{*/
  assert(a._type == b._type);

  if (a._type == LLDouble::LINDOMAIN) {
    if (b._val <= 0.0) {
      ErrorExit(__FILE__, __LINE__, -1, "%f / %f divide by LZERO",
                a._val, b._val);
    }
    return LLDouble(a._val / b._val, LLDouble::LINDOMAIN);
  } else {
    if (b._val <= LLDouble::LSMALL) {
      ErrorExit(__FILE__, __LINE__, -1, "%f / %f (log) divide by LZERO",
                a._val, b._val);
    }
    double z = a._val - b._val;
    return (z <= LLDouble::LSMALL)
      ? LLDouble::LogZero()
      : LLDouble(z, LLDouble::LOGDOMAIN);
  }

}/*}}}*/

std::ostream& operator<<(std::ostream& os, const LLDouble& ref) {/*{{{*/
  os << "LLDouble(" << ref._val << ", ";
  if (ref._type == LLDouble::LOGDOMAIN) os << "log)";
  else os << "lin)";
  return os;
}/*}}}*/



LLDouble::LLDouble(double d, const Type t) {/*{{{*/

  if (t == LINDOMAIN && d < LLDouble::MINLINARG) {
    d = 0.0;
  } else if (t == LOGDOMAIN && d < LSMALL) {
    d = LZERO;
  }
  _val = d;
  _type = t;
}/*}}}*/

LLDouble& LLDouble::to_logdomain() {/*{{{*/
  if (_type == LINDOMAIN) {
    _val = LOG(_val);
    _type = LOGDOMAIN;
  }
  return *this;
}/*}}}*/

LLDouble& LLDouble::to_lindomain() {/*{{{*/
  if (_type == LOGDOMAIN) {
    _val = EXP(_val);
    _type = LINDOMAIN;
  }
  return *this;
}/*}}}*/

const LLDouble& LLDouble::operator=(const LLDouble& ref) {/*{{{*/
  _val = ref._val;
  _type = ref._type;
  return ref;
}/*}}}*/

LLDouble& LLDouble::operator+=(const LLDouble& rhs) {/*{{{*/
  *this = *this + rhs;
  return *this;
}/*}}}*/

LLDouble& LLDouble::operator-=(const LLDouble& rhs) {/*{{{*/
  *this = *this - rhs;
  return *this;
}/*}}}*/

LLDouble& LLDouble::operator*=(const LLDouble& rhs) {/*{{{*/
  *this = *this * rhs;
  return *this;
}/*}}}*/

LLDouble& LLDouble::operator/=(const LLDouble& rhs) {/*{{{*/
  *this = *this / rhs;
  return *this;
}/*}}}*/



double LLDouble::LOG(double a) {/*{{{*/
  if (a < MINLINARG) return LZERO;
  else return log(a);
}/*}}}*/

double LLDouble::EXP(double a) /*{{{*/{
  if (a < MINLOGARG) return 0.0;
  else return exp(a);
}/*}}}*/


