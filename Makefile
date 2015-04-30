# dirs
OBJDIR=objs
SRCDIR=src

# compiler
CXX=mpic++
MPIRUN=mpirun

# compile flags
CXXFLAGS+=-O3 #-std=c++0x

# include flags
INCFLAGS+=$(foreach d, $(VPATH), -I$d)

# link flags
LDFLAGS+=-lpthread -lmpi -lmpi_cxx

# vpath
VPATH = $(SRCDIR) \
	$(SRCDIR)/Config \
	$(SRCDIR)/Data \
	$(SRCDIR)/Master \
	$(SRCDIR)/Model \
	$(SRCDIR)/Model/NeuralNet \
	$(SRCDIR)/SGD \
	$(SRCDIR)/Slave \

	INCDIR=$(VPATH)

# src files
SRCS=\
	$(SRCDIR)/parallelSGD.cpp \
	$(SRCDIR)/Config/Chameleon.cpp \
	$(SRCDIR)/Config/ConfigFile.cpp \
	$(SRCDIR)/Config/confreader.cpp \
	$(SRCDIR)/Data/DataFactory.cpp \
	$(SRCDIR)/Data/TestData.cpp \
	$(SRCDIR)/Data/Mnist.cpp \
	$(SRCDIR)/Data/binary.cpp \
	$(SRCDIR)/Model/model.cpp \
	$(SRCDIR)/SGD/sgd.cpp \
	$(SRCDIR)/SGD/adagrad.cpp \
	$(SRCDIR)/SGD/adadelta.cpp \
	$(SRCDIR)/SGD/kernel_adadelta.cpp \
	$(SRCDIR)/SGD/delayed_adagrad.cpp \
	$(SRCDIR)/SGD/future_adagrad.cpp \
	$(SRCDIR)/SGD/rmsprop.cpp \
	$(SRCDIR)/Master/master.cpp \
	$(SRCDIR)/Slave/slave.cpp \
	$(SRCDIR)/Model/NeuralNet/layer.cpp \
	$(SRCDIR)/Model/NeuralNet/feed_forward_nn.cpp \
	$(SRCDIR)/Model/svm.cpp 

# obj files using patsubst matching
OBJS=$(SRCS:%.cpp=%.o)

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
%.o: %.cpp 
	$(CXX) $(CXXFLAGS) $(INCFLAGS) $(LDFLAGS) $< -c -o $@

# clean
clean:
	rm -rf $(OBJS) parallelSGD
