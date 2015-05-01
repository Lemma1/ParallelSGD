#ifndef _SEQUENCE_DATA_H__
#define _SEQUENCE_DATA_H__

#include "common.h"
#include "confreader.h"
#include "DataFactory.h"

class SequenceData:public DataFactory{
    private:        
        float *m_input;
        float *m_output;

        int m_inputSeqLen;
        int m_outputSeqLen;

        int m_inputDim;
        int m_outputDim;

    public:
        SequenceData(ConfReader *confReader);
        ~SequenceData();

        int getNumberOfData();
        int getDataSize();
        int getLabelSize();
        
        void getDataBatch(float* label, float* data, int* indices, int num);
};

#endif