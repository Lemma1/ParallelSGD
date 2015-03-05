#ifndef __SGD_H__
#define __SGD_H__

class sgdBase
{
public:
    sgdBase() {};
    ~sgdBase() {};

    /* data */
    int m_nParamSize;
    float m_learningRate;

    /* method */
    void virtual updateParams (float *params, float *grad) {};
};

class sgdBasic: public sgdBase
{
public:
    sgdBasic(int paramSize, float learningRate);
    ~sgdBasic();

    /* data */

    /* method */
    void updateParams (float *params, float *grad);
};

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
    float *m_histGradSquare;
};

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

    float *m_EGradSquare;
    float *m_EDeltaSquare;
};

#endif