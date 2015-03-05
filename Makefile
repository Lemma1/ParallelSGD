# dirs
OBJDIR=objs
SRCDIR=tmpsrc
INCDIR=include

# compiler
CXX=mpic++
MPIRUN=mpirun

# compile flags
CXXFLAGS+=-O3 #-std=c++0x

# include flags
INCFLAGS+=-I./$(INCDIR)

# link flags
LDFLAGS+=-lpthread -lmpi -lmpi_cxx -Llib

# src files
SRCS=\
	$(SRCDIR)/Chameleon.cpp \
	$(SRCDIR)/ConfigFile.cpp \
	$(SRCDIR)/MasterConfig.cpp \
	$(SRCDIR)/SlaveConfig.cpp \
	$(SRCDIR)/DataFactory.cpp \
	$(SRCDIR)/TestData.cpp \
	$(SRCDIR)/model.cpp \
	$(SRCDIR)/sgd.cpp \
	$(SRCDIR)/adagrad.cpp \
	$(SRCDIR)/adadelta.cpp \
	$(SRCDIR)/master.cpp \
	$(SRCDIR)/slave.cpp \
	$(SRCDIR)/parallelSGD.cpp

# obj files using patsubst matching
OBJS=$(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRCS))

# nothing to do here
# .PHONY: 

# all comes first in the file, so it is the default 
all : parallelSGD

# run the program
run : parallelSGD
	LD_LIBRARY_PATH=./lib:$(LD_LIBRARY_PATH) $(MPIRUN) -np 2 parallelSGD

# compile main program parallelSGD from all objs 
parallelSGD: $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCFLAGS) $(LDFLAGS) $^ -o $@

# order-only prerequisites for OBJDIR
$(OBJS): | $(OBJDIR)
$(OBJDIR):
	mkdir -p $@

# compile all objs from corresponding %.cpp file and all other *.h files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(INCDIR)/*.h
	$(CXX) $(CPPFLAGS) $(INCFLAGS) $(CXXFLAGS) $< -c -o $@

# clean
clean:
	rm -rf $(OBJDIR) parallelSGD

