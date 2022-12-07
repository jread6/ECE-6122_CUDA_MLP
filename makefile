.SECONDEXPANSION:

EXECUTABLE = nn

MAIN := main.cu

SOURCES = $(wildcard *.cpp)
CU_SOURCES = $(wildcard *.cu)

OBJECTS = $(SOURCES:%.cpp=%.o)
CU_OBJECTS = $(CU_SOURCES:%.cu=%.o)

SFML = -lsfml-graphics -lsfml-window -lsfml-system

CXX := g++
CXXFLAGS := -std=c++0x 

NVCC := nvcc

# CXXFLAGS := -fopenmp -std=c++0x -w	# -w disables warnings
# CXXFLAGS := -std=c++0x -w -I/home/jread6/miniconda3/envs/sfml/include

cuda: 
	$(NVCC) -O3 -DNDEBUG $(CU_SOURCES) -o $(EXECUTABLE)

cpp:
	$(CXX) -O3 -DNDEBUG $(SOURCES)

cuda_dbg: 
	$(NVCC) -g -DDEBUG $(CU_SOURCES) -o $(EXECUTABLE)_debug

release: CXXFLAGS += -O3 -DNDEBUG
release: $(EXECUTABLE)

openmp: CXXFLAGS += -fopenmp -O3
openmp: 
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(EXECUTABLE)_openmp

debug: CXXFLAGS += -g3 -DDEBUG
debug: 
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(EXECUTABLE)_debug

.PHONY: all release debug clean

all: cpp cuda

$(EXECUTABLE): $(OBJECTS)
ifeq ($(EXECUTABLE), executable)
	@echo Edit EXECUTABLE variable in Makefile.
	@echo Using default a.out.
	$(CXX) $(CXXFLAGS) $(OBJECTS)
else
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(EXECUTABLE)
endif

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $*.cpp

clean:
	rm -f $(OBJECTS) $(EXECUTABLE) $(EXECUTABLE)_debug
	rm -Rf *.dSYM

