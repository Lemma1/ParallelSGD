#include <iostream>
#include <fstream>
#include <math.h>

using namespace std;

float generateValue()
{
   return (static_cast<float>(rand()))/RAND_MAX; 
}

void initializeCoef(float* coef, int len)
{
    for (int i=0; i< len; i++)
    {
	coef[i] = generateValue();
    }
}

int main () {
    int numFet = 3;
    int numData = 100;
    ofstream myfile;
    char* buffer;

    //coefficients vector
    float coef[numFet];
    initializeCoef(coef, numFet);
    float tempResult;
    float tempRand;

    ofstream myFile ("TestData.bin", ios::out | ios::binary);

    for (int i=0; i < numData; i++)
    {
	tempResult = 0.0;
	for (int j =0; j < numFet; j++)
	{
	    tempRand =  generateValue();
	    tempResult += tempRand * coef[j];
	    buffer = reinterpret_cast<char*>( &tempRand );
	    myFile.write (buffer, sizeof(float)); 
	}
	buffer = reinterpret_cast<char*>( &tempResult );
	myFile.write (buffer, sizeof(float));
    }
    myFile.close();
    return 0;
}
