#include "DataFactory.h"
#include "TestData.h"

int main()
{
        std::cout <<"1";
	DataFactory* a = new TestData();
	a -> printOutData();
	return 0;
}
