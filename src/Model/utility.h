#include <limits>
#include <cfloat>

float fexp(float x)
{   
	return x< FLT_MIN_EXP ? 0 : exp(x);
};

float fdivide(float *up, float *low)
{	

	if(*low < 0.000001)
	{
		//printf("%f\n", low);
		printf("Warning: zero valued delimiters.\n");
		exit(-1);
		*low = 1;
	}
	return *up / *low;
};

float flog(float x)
{
	if (x == 0) 
	{
		return FLT_MAX/2;
	}
	return log(x);
};

float fmutiplelog(float x, float y)
{
	if (abs(x) < 0.000001)
		return 0;
	if (y < 0.000001)
		return log(0.000001);
	return x*log(y);
};