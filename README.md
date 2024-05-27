# A matrix-free high performance solid solver for coupled fluid-structure interactions
[![Solid mechanics CI](https://github.com/DavidSCN/matrix-free-dealii-precice/actions/workflows/solid_mechanics_ci.yml/badge.svg)](https://github.com/DavidSCN/matrix-free-dealii-precice/actions/workflows/solid_mechanics_ci.yml)
[![v9.4](https://github.com/DavidSCN/matrix-free-dealii-precice/actions/workflows/backward_compatibility.yml/badge.svg)](https://github.com/DavidSCN/matrix-free-dealii-precice/actions/workflows/backward_compatibility.yml)

This project provides matrix-free high-performance solver for coupled simulations: a hyper-elastic `solid` solver for coupled fluid-structure interactions, which was primarily developed on the basis of the ['large-strain-matrix-free'](https://github.com/davydden/large-strain-matrix-free) project, and a `heat` solver describing the thermal diffusion in a rigid body. The startin point of this project was given by the matrix-based coupled codes in the ['dealii-adapter'](https://github.com/precice/dealii-adapter) project. Special thanks goes to [@peterrum](https://github.com/peterrum/) for his valuable deal.II advice at all times.

## Description
The program builds on the [finite-element library deal.II ](https://github.com/dealii/dealii) and the [coupling library preCICE](https://github.com/precice/precice) and includes the following capabilities:
- [x] matrix-free
- [x] geometric multigrid preconditioner
- [x] fully implicit coupling
- [ ] subcycling
- [x] arbitrary number of interface nodes
- [x] selectable interface node location
- [x] mpi parallelism and vectorization

The solid mechanics solver:
- [x] nonlinear hyperelastic neo-Hookean material
- [x] Newton-Raphson method
- [x] Newmark time integration

The heat diffusion solver:
- [x] backward Euler time integration


## Installation
In order to build the program, both libraries (deal.II and preCICE) need to be installed on your system:

### Step 1: install deal.II
At least version `9.4` (`9.3` might work as well, but this version in not tested any more) or greater is required. Compatibility for older deal.II versions and corresponding older repository versions might be available thorugh the [list of releases](https://github.com/DavidSCN/matrix-free-dealii-precice/releases). You can use the following command line instructions in order to download and compile deal.II. Note that the library relies on [p4est](https://www.p4est.org/) in order to handle distributed meshes and you need to adjust the `P4EST_DIR` according to your installation. If you have not yet installed p4est, you might want to download the latest tarball and run the `p4est-setup.sh` script located in your deal.II directory at `dealii/doc/external-libs`. In order to compile p4est properly, the script might require you to export the mpi wrapper before starting it: `export CC=mpicc` and `export CXX=mpicxx`. On Linux-based systems, the [candi](https://github.com/dealii/candi) (compile & install) script offers help for the installation process. A complete installation guide including all installation options is also given on the [deal.II installation page](https://dealii.org/developer/readme.html#installation).
```
git clone https://github.com/dealii/dealii.git
mkdir build && cd build

cmake \
    -D CMAKE_BUILD_TYPE="DebugRelease" \
    -D CMAKE_CXX_FLAGS="-march=native -std=c++17" \
    -D DEAL_II_CXX_FLAGS_RELEASE="-O3" \
    -D DEAL_II_WITH_MPI="ON" \
    -D DEAL_II_WITH_P4EST="ON" \
    -D P4EST_DIR="../path/to/p4est/installation" \
    -D DEAL_II_WITH_LAPACK="ON" \
    -D DEAL_II_WITH_HDF5="OFF" \
    -D DEAL_II_WITH_MUPARSER="ON" \
    -D DEAL_II_COMPONENT_EXAMPLES="OFF" \
    ..

make -j8
```

### Step 2: install preCICE
At least version `3.0` or greater is required. A nice overview of various installation options is given on the [preCICE installation page](https://www.precice.org/installation-overview.html).

### Step 3: build the program
Similar to the deal.II programs, run
```
mkdir build && cmake -DDEAL_II_DIR=/path/to/deal.II -Dprecice_DIR=/path/to/precice ..
```
in this directory to configure the problem. The explicit specification of the library locations can be omitted if they are installed globally or via environment variables. You can then configure the program in debug mode or release mode by calling either `make debug` or `make release`. Afterwards, run `make solid` in order to build the solid mechanics FSI solver, `make heat` in order to build the thermal diffusion heat solver or just `make` in order to build both solvers. Exemplary parameter files for each individual solver can be found in the root directory of this repository (`elasticity.prm` for the FSI solver, `heat_transfer.prm` for the heat solver) or in the `./examples` directory, which contains application examples. Note that the polynomial degree and the applied quadrature formula (both can be specified in the parameter files) are templated within the program. By default, common polynomial degrees (1-6) and corresponding common quadrature orders are supported. However, both template parameters can be modified in the `./include/base/fe_integrator.h` file. More information about the templating of these parameters and their selection can be found in the corresponding [deal.II documentation](https://dealii.org/developer/doxygen/deal.II/classFEEvaluation.html).

## How to run a simulation ?
Compiling the program generates two executables: `solid` and `heat`. In order to run a simulation, add the location of your executable to your `PATH` environment variable or copy the executable directly into the simulation directory.  Afterwards, configure your case using the parameter file, e.g. `elasticity.prm` and start the case, e.g. using four processes, by running

```
mpirun -np 4 ./solid elasticity.prm
```

## How to visualize results ?
The programs generates `vtu` output files after every `Output tick` of simulation time and stores them in the `Output folder`, which can be specified in the parameter file. The `vtu` format can be visualized with common visualization software supporting `vtk` files such as `ParaView` or `VisIt`. However, the results are written using [`higher_order_cells`](https://dealii.org/developer/doxygen/deal.II/structDataOutBase_1_1VtkFlags.html#aa9dd1830c0ff35a2431704c4d45453eb) by default and a sufficiently recent version of these software packages supporting the higher-order feature is required to visualize the files (e.g. starting from`Paraview` version 5.5.0). If necessary, writing higher-order cells can easily be disabled in the source code.


## How to add a test case ?
All test cases are located in the (sub)directory `./include/cases/`. You can add your own case by copying one of the existing cases or editing the `./include/cases/case_template.h` file. Have a look at the description in the respective files for further information. Afterwards, you need to add your new case to the `./include/cases/case_selector.h` and recompile the program.

## I still have open questions
If you still have questions, you can ask them preferably on [GitHub discussions](https://github.com/DavidSCN/matrix-free-dealii-precice/discussions) here or alternatively in one of the [preCICE community channels](https://precice.org/community-channels.htm).
