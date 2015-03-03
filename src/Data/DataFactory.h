#ifndef DATAFACTORY_H__
#define DATAFACTORY_H__

#include <iostream>

class DataFactory
{
    public:
	DataFactory();
	//DataFactory(int);
	virtual int getNumberOfData() {return 0;};
	virtual void printOutData() {};
	virtual void getDataBatch(float*, float*, int*, int) {};
    protected:
	int numFet;
	int numData;
	std::string dataName;
	virtual float getDataByIndex(int,int) {return 0.0;};
};

#endif
