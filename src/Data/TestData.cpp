#include <iostream>
#include <fstream>

#include "TestData.h"

TestData::TestData()
{
    numFet = 3; 
    numData = 100;
    dataName = "TestData.bin";
    dataVector = new float[numData * (numFet + 1)];
    loadData();
}

TestData::~TestData()
{
    delete [] dataVector;
}

void TestData::printOutData()
{
    std::ifstream ifs( dataName.c_str(), std::ios::binary);
    float read;
    for (int i=0; i < numData; i++)
    {
	for (int j=0; j < numFet; j++)
	{
	    ifs.read( reinterpret_cast<char*> (&read), sizeof(float) );
	    std::cout << read << ",";
	}
	    ifs.read( reinterpret_cast<char*> (&read), sizeof(float) );
	    std::cout << read << std::endl;
    }
    ifs.close();
}

void TestData::loadData()
{
    std::ifstream ifs( dataName, std::ios::binary);
    float read;
    for (int i=0; i < numData; i++)
    {
	for (int j=0; j < numFet + 1; j++)
	{
	    ifs.read( reinterpret_cast<char*> (&read), sizeof(float) );
	    dataVector[i*(numFet+1)+j] = read;
	}
    }
    ifs.close();
}

int TestData::getNumberOfData()
{
    return numData;
}

//The first int is for data index, second int is for feature index
float TestData::getDataByIndex(int dataIndex, int fetIndex)
{
   return(dataVector[ dataIndex * (numFet+1) + fetIndex ]);
}

void TestData::getDataBatch(float* label, float* data, int* indexs, int num)
{
    for (int i=0; i< num; i++)
    {
    	for (int j=0; j< numFet; j++)
    	{     
    	    data[i*numFet+j] = getDataByIndex(indexs[i],j);
    	}
        label[i] = getDataByIndex(indexs[i], numFet);
    }
    
}


