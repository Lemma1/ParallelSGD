#include "DataFactory.h"
#include "TestData.h"
#include "Mnist.h"
int main()
{
	//DataFactory* a = new TestData();
    Mnist* a = new Mnist(1);
    //a ->printLabel();
	a -> printOutDataFromData();
	return 0;
}
