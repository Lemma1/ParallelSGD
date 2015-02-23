# dirs
OBJDIR=objs
SRCDIR=src

# compiler
CXX=mpic++
MPIRUN=mpirun

# compile flags
CXXFLAGS+=-O3 -std=c++0x

# dynamic link flags
LDFLAGS+=-lpthread -lmpi -lmpi_cxx -Llib

# src files
SRCS=\
     $(SRCDIR)/main.cpp \
     $(SRCDIR)/....           # *** TO ADD MORE ***

# obj files using patsubst matching
OBJS=$(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRCS))

# nothing to do here
.PHONY: 

# all comes first in the file, so it is the default 
all : parallelSGD

# run the program
run : parallelSGD
	LD_LIBRARY_PATH=./lib:$(LD_LIBRARY_PATH) $(MPIRUN) -np 4 parallelSGD

# compile main program parallelSGD from all objs 
parallelSGD: $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

# order-only prerequisites for OBJDIR
$(OBJS): | $(OBJDIR)
$(OBJDIR):
	mkdir -p $@

# compile all objs from corresponding %.cpp file and all other *.h files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(SRCDIR)/*.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -c -o $@

# clean
clean:
	rm -rf $(OBJDIR) parallelSGD

