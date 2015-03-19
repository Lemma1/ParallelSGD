# dirs
OBJDIR=objs
SRCDIR=src
INCDIR=$(VPATH)

# compiler
CXX=mpic++
MPIRUN=mpirun

# compile flags
CXXFLAGS+=-O3 #-std=c++0x

# include flags
INCFLAGS+=$(foreach d, $(VPATH), -I$d)

# link flags
LDFLAGS+=-lpthread -lmpi -lmpi_cxx -Llib

# vpath
VPATH = $(SRCDIR) \
	$(SRCDIR)/Config \
	$(SRCDIR)/Data \
	$(SRCDIR)/Master \
	$(SRCDIR)/Model \
	$(SRCDIR)/SGD \
	$(SRCDIR)/Slave \
	
# src files
SRCS=\
	Chameleon.cpp \
	ConfigFile.cpp \
	MasterConfig.cpp \
	SlaveConfig.cpp \
	DataFactory.cpp \
	TestData.cpp \
	model.cpp \
	sgd.cpp \
	adagrad.cpp \
	adadelta.cpp \
	rmsprop.cpp \
	master.cpp \
	slave.cpp \
	parallelSGD.cpp\
	svm.cpp

# obj files using patsubst matching
OBJS=$(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRCS))

# nothing to do here
# .PHONY: 

# all comes first in the file, so it is the default 
all : parallelSGD

# run the program
run : parallelSGD
	LD_LIBRARY_PATH=./lib:$(LD_LIBRARY_PATH) $(MPIRUN) -np 4 parallelSGD

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

