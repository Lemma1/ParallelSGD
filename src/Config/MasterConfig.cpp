#include "MasterConfig.h"

int getMasterIntConf(std::string key)
{
   int confValue = cf.Value(MasterKey, key);
   return confValue; 
}

float getMasterFloatConf(std::string key)
{
   float confValue = cf.Value(MasterKey, key);
   return confValue; 
}

std::string getMasterStringConf(std::string key)
{
   std::string confValue = cf.Value(MasterKey, key);
   return confValue; 
}
