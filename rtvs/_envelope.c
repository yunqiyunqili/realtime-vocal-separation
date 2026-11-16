#include <math.h>
void envelope(const double*x,double*y,long n,double attack,double release) {
 double state=0.;for(long i=0;i<n;i++){double a=x[i]>state?attack:release;state=a*state+(1.-a)*x[i];y[i]=state;}
}
