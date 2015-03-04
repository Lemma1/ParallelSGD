#include "MasterConfig.h"

#include <iostream>

int main() {
    std::cout << getMasterIntConf("number of Slave") << std::endl;
    std::cout << getMasterStringConf("foo") << std::endl;
    std::cout << getMasterFloatConf("shit") << std::endl;
    return 0;
}
