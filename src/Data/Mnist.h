

#include <iostream>

#include "DataFactory.h"

class Mnist:public DataFactory{
     private:
         std::string labelName; //the path of label file 
         float* dataVector;
         void loadData();
     protected:
         float getDataByIndex(int,int);
     public:
         Mnist(int);
         ~Mnist();
         int getNumberOfData();
         int getDataSize();
         int getLabelSize();

         void getDataBatch(float*,float*,int*,int);

         //for test&debug
         void printLabel();
         void printOutDataFromFile();
         void printOutDataFromData();
};
