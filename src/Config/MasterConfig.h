#ifndef MASTERCONFIG_H__
#define MASTERCONFIG_H__

#include "ConfigFile.h"

#include <string>

static ConfigFile cf("config.conf");
static std::string MasterKey = "Master";

int getMasterIntConf(std::string);
std::string getMasterStringConf(std::string);
float getMasterDoubleConf(std::string);

#endif
