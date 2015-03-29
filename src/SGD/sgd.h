#ifndef __SGD_H__
#define __SGD_H__

#include <stdio.h>
#include "confreader.h"

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
    int m_useMomentum;
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
    sgdBasic(ConfReader *confReader, int paramSize);
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
    adagrad(ConfReader *confReader, int paramSize);
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
    adadelta(ConfReader *confReader, int paramSize);
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
    rmsprop(ConfReader *confReader, int paramSize);
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