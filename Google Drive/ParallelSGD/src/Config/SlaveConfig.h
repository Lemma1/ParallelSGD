#ifndef SLAVECONFIG_H__
#define SLAVECONFIG_H__

#include "ConfigFile.h"

#include <string>

#ifndef __CONFIG_
#define __CONFIG_
static ConfigFile cf("config.conf");
#endif

static std::string SlaveKey = "Slave";


int getSlaveIntConf(std::string);
std::string getSlaveStringConf(std::string);
float getSlaveFloatConf(std::string);

#endif
