# A matrix-free high performance solid solver for coupled fluid-structure interactions
![GitHub CI](https://github.com/DavidSCN/matrix-free-dealii-precice/workflows/GitHub%20CI/badge.svg)

This project provides a matrix-free high performance solid solver for coupled fluid-structure interactions. It was primarily developed on the basis of the ['large-strain-matrix-free'](https://github.com/davydden/large-strain-matrix-free) project and the ['dealii-adapter'](https://github.com/precice/dealii-adapter) project.

## Description
The program builds on the [finite-element library deal.II ](https://github.com/dealii/dealii) and the [coupling library preCICE](https://github.com/precice/precice) and includes the following capabilities:
- [x] nonlinear hyperelastic neo-Hookean material
- [x] Newton-Raphson method
- [x] matrix-free
- [x] geometric multigrid preconditioner
- [x] mpi parallelism and vectorization
- [x] Newmark time integration
- [x] fully implicit coupling
- [ ] subcycling
- [x] arbitrary number of interface nodes
- [x] selectable interface node location


## Installation
In order to build the program, both libraries (deal.II and preCICE) need to be installed on your system:

### Step 1: install deal.II
At least version `9.2` or greater is required. Older versions might work as well, but have not been tested. You can use the following command line instructions in order to download and compile deal.II. Note that the library relies on [p4est](https://www.p4est.org/) in order to handle distributed meshes and you need to adjust the `P4EST_DIR` according to your installation. If you have not yet installed p4est, you might want to download the [latest tarball](https://p4est.github.io/release/p4est-2.2.tar.gz) and run the `p4est-setup.sh` script located in your deal.II directory at `dealii/doc/external-libs`. A complete installation guide including all installation options is also given on the [deal.II installation page](https://dealii.org/developer/readme.html#installation). 
```
git clone https://github.com/dealii/dealii.git
mkdir build && cd build

cmake \
    -D CMAKE_BUILD_TYPE="DebugRelease" \
    -D CMAKE_CXX_FLAGS="-march=native -std=c++17" \
    -D DEAL_II_CXX_FLAGS_RELEASE="-O3" \
    -D DEAL_II_WITH_MPI="ON" \
    -D DEAL_II_WITH_P4EST="ON" \
    -D P4EST_DIR="../path/to/p4est/installation"
    -D DEAL_II_WITH_LAPACK="ON" \
    -D DEAL_II_WITH_HDF5="OFF" \
    -D DEAL_II_WITH_MUPARSER="ON" \
    -D DEAL_II_WITH_THREADS="ON" \
    -D DEAL_II_COMPONENT_EXAMPLES="OFF" \
    ../dealii

make -j8
```

### Step 2: install preCICE
At least version 2.0 or greater is required. A nice overview of various installation options is given on the [preCICE installation page](https://www.precice.org/installation-overview.html). 

### Step 3: build the program
Similar to deal.II, run
```
mkdir build && cmake -DDEAL_II_DIR=/path/to/deal.II -Dprecice_DIR=/path/to/precice ..
```
in this directory to configure the problem. The explicit specification of the library locations can be omitted if they are installed globally or via environment variables. You can then compile the program in debug mode or release mode by calling either `make debug` or `make release`. Note that the polynomial degree and the applied quadrature formula are templated within the program. By default, the polynomial degrees 1 to 4 and a quadrature order of 'degree + 1' are compiled. The list of template paramter combinations can be found (and modified) at `./include/template_list.h`.

## How to run a simulation ?
Compiling the program generates an executable called `solid`. In order to run a simulation, add the location of your executable to your `PATH` environment variable or copy the executable `solid` directly into the simulation directory.  Afterwards, configure your case using the parameter file `parameters.prm` and start the case, e.g. using four processes, by running
```
mpirun -np 4 ./solid parameters.prm
```
## How to add a test case ? 
All test cases are located in the directory `./include/cases/`. You can add your own case by copying one of the existing cases or editing the `./include/cases/case_template.h` file. Have a look at the description in the respective files for further information. Afterwards, you need to add your new case to the `./include/cases/case_selector.h` and recompile the program.
