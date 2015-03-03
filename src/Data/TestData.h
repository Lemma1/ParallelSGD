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
    public:
	TestData();
	~TestData();
	int getNumberOfData();
	void printOutData();
	float getDataByIndex(int, int);
	void getDataBatch(float* data, int* indexs, int num);
};

#endif
