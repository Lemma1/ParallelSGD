//for Mnist dats set 
#include <iostream>
#include <fstream>

#include "Mnist.h"

Mnist::Mnist(int ifTrain){
    numFet = 28*28;
    if(ifTrain){
        dataName = "train-images-idx3-ubyte";//the train data:
        labelName = "train-labels-idx1-ubyte";
        numData = 60000;
    }
    else{
        dataName = "t10k-images-idx3-ubyte";
        labelName = "t10k-labels-idx1-ubyte";
        numData = 5000;
    }
    dataVector = new unsigned int[numData * (numFet + 1)];
    memset(dataVector,0,numData*(numFet+1));
    loadData();
}

Mnist::~Mnist(){
    delete [] dataVector;
}

int Mnist::getNumberOfData(){
    return numData;
}

unsigned int Mnist::getDataByInde(int dataIndex, int fetIndex)
{
   return(dataVector[ dataIndex * (numFet+1) + fetIndex ]);
}

unsigned int Mnist::swap(unsigned int d)
{
    unsigned int a;
    unsigned char *dst = (unsigned char *)&a;
    unsigned char *src = (unsigned char *)&d;
    dst[0] = src[7];
    dst[1] = src[6];
    dst[2] = src[5];
    dst[3] = src[4];
    dst[4] = src[3];
    dst[5] = src[2];
    dst[6] = src[1];
    dst[7] = src[0];
    return a;
}

void Mnist::loadData(){
    unsigned int temp;
    uint8_t temp8;
    std::ifstream mnFer(dataName,std::ios::binary);
    std::ifstream mnLab(labelName,std::ios::binary);
    for(int i=0;i<2;i++){
        mnLab.read(reinterpret_cast<char*>(&temp),sizeof(int));
        mnFer.read(reinterpret_cast<char*>(&temp),sizeof(int));
        mnFer.read(reinterpret_cast<char*>(&temp),sizeof(int));
    }//skip the first two lines 
    for (int i=0;i<numData;i++){
        for(int j=0;j<=numFet;j++){
            if(j==numFet){
                mnLab.read(reinterpret_cast<char*>(&temp8),sizeof(char));
                temp = temp8;
                dataVector[i*(numFet+1)+j] =(temp);
            }
            else{
                mnFer.read(reinterpret_cast<char*>(&temp8),sizeof(char));
                temp = temp8;
                dataVector[i*(numFet+1)+j] =(temp);
            }
        }
    }
    mnFer.close();
    mnLab.close();
}

void Mnist::getDataBatch(float* label, float* data, int* indexs, int num)
{
    for (int i=0; i< num; i++)
    {
        for (int j=0; j< numFet; j++)
        {
            data[i*numFet+j] = getDataByIndex(indexs[i],j);
            label[i] = getDataByIndex(indexs[i], numFet);
        }
    }
}

void Mnist::printOutDataFromFile()
{
    std::ifstream ifs( dataName, std::ios::binary);
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

void Mnist::printOutDataFromData(){
    for (int i=0;i<numData;i++){
        for(int j=0;j<numFet+1;j++){
            std::cout << dataVector[i*(numFet+1)+j]<<",";
        }
        std::cout<<std::endl;
    }
}


void Mnist::printLabel(){
    for(int i=1;i<numData+1;i++){
        std::cout << dataVector[i*(numFet+1)-1]<<std::endl;
    }
    std::cout << numFet <<numData << std::endl;
}

