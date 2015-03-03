#include "DataFactory.h"
#include "TestData.h"

int main()
{
        std::cout <<"1";
	DataFactory* a = new TestData();
	a -> printOutData();
	std::cout << a -> getDataByIndex(20,3);
	return 0;
}
