#include "binary.h"
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>

BinaryData::BinaryData()
{
    numFet = 123;
    numData = 32561;
    dataName = "a9a";
    dataVector = new float[numData * (numFet + 1)];
    memset(dataVector,0x00,numData*(numFet+1));
    loadData();
}

BinaryData::~BinaryData()
{
    delete [] dataVector;
}

void BinaryData::loadData()
{
    int* templist = new int[2];
    int linecounter = 0;
    int offset;
    std::ifstream bn(dataName.c_str(), std::ios::in);
    if (bn.good())
    {
	std::string line;
		while(std::getline(bn, line)) 
	{
	    offset = linecounter * (numFet + 1);
	    std::istringstream ss(line);
	    std::string word;
	    ss >> word;
	    dataVector[offset + numFet] = atoi(word.c_str());
	    while(ss >> word)
	    {
		parseWord(templist, word);
		//std::cout << templist[0] << std::endl;
		dataVector[offset + templist[0]-1] = templist[1];
	    }
	    linecounter++ ;
	}
    }

}

int BinaryData::getNumberOfData()
{
    return numData;
}

void BinaryData::printOutData()
{
    for (int i=0;i<numData;i++){
        for(int j=0;j<numFet+1;j++){
            std::cout << dataVector[i*(numFet+1)+j]<<",";
        }
        std::cout<<std::endl;
    }
}

float BinaryData::getDataByIndex(int dataIndex, int fetIndex)
{
   return(dataVector[ dataIndex * (numFet+1) + fetIndex ]);
}

void BinaryData::getDataBatch(float* label, float* data, int* indexs, int num)
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

void BinaryData::parseWord(int* list, std::string word)
{
    std::istringstream ss(word);
    std::string token;
    int counter = 0;
    while(std::getline(ss, token, ':'))	    
    {
	//std::cout << token << '\n';
	list[counter] = atoi(token.c_str());
	counter++;
    }
}
