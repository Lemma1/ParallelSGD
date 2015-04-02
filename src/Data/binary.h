#ifndef __BINARY__H
#define __BINARY__H

#include <iostream>

#include "DataFactory.h"

class BinaryData
    : public DataFactory
{
    public:
	BinaryData();
	~BinaryData();
	int getNumberOfData();
	void printOutData();
	void getDataBatch(float*, float*, int*, int);    
    private:
	float* dataVector;
	void loadData();
	void parseWord(int*, std::string);
	float getDataByIndex(int, int);
};

#endif
