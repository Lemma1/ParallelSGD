#include "MasterConfig.h"
#include "SlaveConfig.h"

#include <iostream>

int main() {
    std::cout << getMasterIntConf("number of Slave") << std::endl;
    std::cout << getMasterStringConf("foo") << std::endl;
    std::cout << getMasterDoubleConf("shit") << std::endl;
    std::cout << getSlaveIntConf("number of Slave") << std::endl;
    return 0;
}
