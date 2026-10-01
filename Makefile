# Сборка четырёх программ из двух исходников:
#   poisson_seq — последовательная (poisson_omp.cpp без OpenMP)
#   poisson_omp — OpenMP
#   poisson_mpi — MPI без OpenMP
#   poisson_hyb — MPI + OpenMP
#
# Компилятор выбирается переменной COMPILER, по умолчанию — по машине:
#   xl    — Polus (ppc64le): IBM XL, перед сборкой `module load SpectrumMPI`
#   gnu   — GCC (на Polus тоже есть, 4.8.5)
#   clang — macOS: Apple clang + libomp из Homebrew, Open MPI из Homebrew
#
#   make                    # всё
#   make COMPILER=gnu       # явно GCC
#   make clean

ARCH := $(shell uname -m)
OS   := $(shell uname -s)
ifeq ($(ARCH),ppc64le)
  COMPILER ?= xl
else ifeq ($(OS),Darwin)
  COMPILER ?= clang
else
  COMPILER ?= gnu
endif

ifeq ($(COMPILER),xl)
  CXX      := xlC_r
  MPICXX   := mpixlC
  # -qstrict: не переставлять операции с плавающей точкой, чтобы результат
  # не зависел от уровня оптимизации. -qhot не используем: по документации
  # Polus он делает программу недетерминированной.
  CXXFLAGS := -O3 -qstrict -qarch=pwr8 -qtune=pwr8
  OMPFLAGS := -qsmp=omp
  MPIENV   :=
else ifeq ($(COMPILER),clang)
  LIBOMP   ?= $(shell brew --prefix libomp 2>/dev/null)
  CXX      := clang++
  MPICXX   := mpicxx
  CXXFLAGS := -O3 -std=c++11 -Wall -Wextra
  OMPFLAGS := -Xpreprocessor -fopenmp -I$(LIBOMP)/include -L$(LIBOMP)/lib -lomp
  MPIENV   :=
else
  CXX      := g++
  MPICXX   := mpicxx
  CXXFLAGS := -O3 -std=c++11 -Wall -Wextra
  ifeq ($(ARCH),ppc64le)
    CXXFLAGS += -mcpu=power8
  endif
  OMPFLAGS := -fopenmp
  # Spectrum MPI / Open MPI: обёртка mpicxx вызывает компилятор из OMPI_CXX.
  MPIENV   := OMPI_CXX=g++
endif

HDR := src/problem.h src/solver.h src/common.h

all: poisson_seq poisson_omp poisson_mpi poisson_hyb

poisson_seq: src/poisson_omp.cpp $(HDR)
	$(CXX) $(CXXFLAGS) -o $@ $<

poisson_omp: src/poisson_omp.cpp $(HDR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

poisson_mpi: src/poisson_mpi.cpp $(HDR)
	$(MPIENV) $(MPICXX) $(CXXFLAGS) -o $@ $<

poisson_hyb: src/poisson_mpi.cpp $(HDR)
	$(MPIENV) $(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

clean:
	rm -f poisson_seq poisson_omp poisson_mpi poisson_hyb

.PHONY: all clean
