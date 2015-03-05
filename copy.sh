#/bin/bash

for cpp in `(find src/* | grep ".cpp$")`
    do cp $cpp tmpsrc/; 
done

for head in `(find src/* | grep ".h$")` 
    do cp $head include/; 
done
