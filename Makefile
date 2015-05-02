# dirs
OBJDIR=objs
SRCDIR=src
LIBDIR=lib
ROOTDIR=$(shell pwd)
UNAME=$(shell uname)

# compiler
CXX=mpic++
MPIRUN=mpirun

# compile flags
CXXFLAGS+=-O3#-std=c++0x
ifeq ($(UNAME), Linux)
    CXXFLAGS+=-mavx
endif

# include flags
INCFLAGS+=$(foreach d, $(VPATH), -I$d)
INCFLAGS+=-I$(LIBDIR)/openblas/include

# link flags
LDFLAGS+=-lpthread -lmpi -lmpi_cxx -lopenblas
LDFLAGS+=-L$(LIBDIR)/
LDFLAGS+=-L$(LIBDIR)/openblas/lib

# vpath
VPATH = $(SRCDIR) \
	$(SRCDIR)/Config \
	$(SRCDIR)/Data \
	$(SRCDIR)/Master \
	$(SRCDIR)/Model \
	$(SRCDIR)/Model/NeuralNet \
	$(SRCDIR)/Model/RNN/connection \
	$(SRCDIR)/Model/RNN/helper \
	$(SRCDIR)/Model/RNN/layer \
	$(SRCDIR)/Model/RNN/network \
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
	$(SRCDIR)/Data/sequence_data.cpp \
	$(SRCDIR)/Model/model.cpp \
	$(SRCDIR)/SGD/sgd.cpp \
	$(SRCDIR)/SGD/adagrad.cpp \
	$(SRCDIR)/SGD/adadelta.cpp \
	$(SRCDIR)/SGD/kernel_adadelta.cpp \
	$(SRCDIR)/SGD/delayed_adagrad.cpp \
	$(SRCDIR)/SGD/future_adagrad.cpp \
	$(SRCDIR)/SGD/delayed_adadelta.cpp \
	$(SRCDIR)/SGD/rmsprop.cpp \
	$(SRCDIR)/Master/master.cpp \
	$(SRCDIR)/Slave/slave.cpp \
	$(SRCDIR)/Model/NeuralNet/layer.cpp \
	$(SRCDIR)/Model/NeuralNet/feed_forward_nn.cpp \
	$(SRCDIR)/Model/RNN/connection/rnn_connection.cpp \
	$(SRCDIR)/Model/RNN/helper/matrix.cpp \
	$(SRCDIR)/Model/RNN/helper/nonlinearity.cpp \
	$(SRCDIR)/Model/RNN/layer/rnn_layer.cpp \
	$(SRCDIR)/Model/RNN/layer/rnn_input_layer.cpp \
	$(SRCDIR)/Model/RNN/layer/rnn_mse_layer.cpp \
	$(SRCDIR)/Model/RNN/layer/rnn_softmax_layer.cpp \
	$(SRCDIR)/Model/RNN/layer/lstm_layer.cpp \
	$(SRCDIR)/Model/RNN/network/rnn_lstm.cpp \
	$(SRCDIR)/Model/RNN/network/rnn_translator.cpp \
	$(SRCDIR)/Model/svm.cpp 

# obj files using patsubst matching
OBJS=$(SRCS:%.cpp=%.o)

# nothing to do here
# .PHONY: 

# all comes first in the file, so it is the default 
all : parallelSGD

# run the program
run : parallelSGD
	LD_LIBRARY_PATH=./$(LIBDIR):./$(LIBDIR)/openblas/lib:$(LD_LIBRARY_PATH) $(MPIRUN) -np 2 parallelSGD

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
