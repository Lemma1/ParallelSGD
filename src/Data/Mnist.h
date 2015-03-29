

#include <iostream>

#include "DataFactory.h"

class Mnist:public DataFactory{
     private:
         std::string labelName; //the path of label file 
         unsigned int* dataVector;
         void loadData();
     protected:
         unsigned int getDataByInde(int,int);
         unsigned int swap(unsigned int);
     public:
         Mnist(int);
         ~Mnist();
         int getNumberOfData();
         void getDataBatch(float*,float*,int*,int);

         //for test&debug
         void printLabel();
         void printOutDataFromFile();
         void printOutDataFromData();
};
