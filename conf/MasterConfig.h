#ifndef MASTERCONFIG_H__
#define MASTERCONFIG_H__

#include "ConfigFile.h"

#include <string>

#ifndef __CONFIG_
#define __CONFIG_
static ConfigFile cf("config.conf");
#endif
static std::string MasterKey = "Master";


int getMasterIntConf(std::string);
std::string getMasterStringConf(std::string);
double getMasterDoubleConf(std::string);

#endif
