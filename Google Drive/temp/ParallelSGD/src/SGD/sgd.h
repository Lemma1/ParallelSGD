#ifndef __SGD_H__
#define __SGD_H__

class sgdBase
{
public:
    sgdBase();
    ~sgdBase();

    /* data */
    int paramSize;
    float learningRate;

    /* method */
    void virtual updateParams (float *params, float *grad);
};

class sgdBasic: public sgdBase
{
public:
    sgdBasic(int size, float eta);
    ~sgdBasic();

    /* data */

    /* method */
    void updateParams (float *params, float *grad);
};

#endif