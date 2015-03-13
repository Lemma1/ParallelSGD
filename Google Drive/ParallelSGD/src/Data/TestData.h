#ifndef TESTDATA_H__
#define TESTDATA_H__

#include <iostream>
#include <fstream>

#include "DataFactory.h"

class TestData 
    :public DataFactory
{
    private:
	float* dataVector;
	void loadData();
    protected:
	float getDataByIndex(int, int);
    public:
	TestData();
	~TestData();
	int getNumberOfData();
	void printOutData();
	void getDataBatch(float*, float*, int*, int);
};

#endif
