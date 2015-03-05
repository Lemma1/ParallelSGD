#ifndef __SGD_H__
#define __SGD_H__

#include <stdio.h>

class sgdBase
{
public:
    sgdBase() {};
    ~sgdBase() {};

    /* data */

    /* method */
    void virtual updateParams (float *params, float *grad) {};
protected:
    /* data */
    int m_nParamSize;
    float m_learningRate;

    /* method */
    //TODO void truncate (float);
};

/****************************************************************
* BASIC SGD
****************************************************************/
class sgdBasic: public sgdBase
{
public:
    sgdBasic(int paramSize, float learningRate);
    ~sgdBasic();

    /* data */

    /* method */
    void updateParams (float *params, float *grad);
};

/****************************************************************
* ADAGRAD
****************************************************************/
class adagrad: public sgdBase
{
public:
    adagrad(int paramSize, float learningRate);
    ~adagrad();

    /* data */

    /* method */
    void updateParams (float *params, float *grad);

private:
    /* data */
    float *m_histSquareGrad;
};

/****************************************************************
* ADADELTA
****************************************************************/
class adadelta: public sgdBase
{
public:
    adadelta(int paramSize, float decayFactor, float stableConst);
    ~adadelta();

    /* data */

    /* method */
    void updateParams (float *params, float *grad);

private:
    /* data */
    float m_decayFactor;
    float m_stableConst;

    float *m_ESquareGrad;
    float *m_ESquareDelta;
};

/****************************************************************
* RMSPROP
****************************************************************/
class rmsprop: public sgdBase
{
public:
    rmsprop(int paramSize, float decayFactor);
    ~rmsprop();

    /* data */

    /* method */
    void updateParams (float *params, float *grad);

private:
    /* data */
    float m_decayFactor;

    float *m_meanSquareGrad;    
};

#endif