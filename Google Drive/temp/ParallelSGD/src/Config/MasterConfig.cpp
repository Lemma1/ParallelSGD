#include "MasterConfig.h"

int getMasterIntConf(std::string key)
{
   int confValue = cf.Value(MasterKey, key);
   return confValue; 
}

double getMasterDoubleConf(std::string key)
{
   double confValue = cf.Value(MasterKey, key);
   return confValue; 
}

std::string getMasterStringConf(std::string key)
{
   std::string confValue = cf.Value(MasterKey, key);
   return confValue; 
}
