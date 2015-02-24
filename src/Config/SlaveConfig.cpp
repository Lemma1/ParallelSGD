#include "SlaveConfig.h"

int getSlaveIntConf(std::string key)
{
   int confValue = cf.Value(SlaveKey, key);
   return confValue; 
}

double getSlaveDoubleConf(std::string key)
{
   float confValue = cf.Value(SlaveKey, key);
   return confValue; 
}

std::string getSlaveStringConf(std::string key)
{
    std::string confValue = cf.Value(SlaveKey, key);
   return confValue; 
}
