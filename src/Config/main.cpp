#include <stdio.h>
#include "confreader.h"

int main () {
	ConfReader *confReader = new ConfReader("config.conf", "Master");
	int i = confReader->getInt("number of Slave");
	printf("%d\n",i);
	return 0;
}