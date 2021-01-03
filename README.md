# A matrix-free high performance solid solver for coupled fluid-structure interactions
![Building](https://github.com/DavidSCN/matrix-free-dealii-precice/workflows/Building/badge.svg)

## Installation
The program builds on the [finite-element library deal.II ](https://github.com/dealii/dealii) and the [coupling library preCICE](https://github.com/precice/precice). Both libraries need to be installed on your system:

### Step 1: install deal.II
The program requires at least version `9.2` or greater. Older versions might work as well, but have not been tested. You can use the following command line instructions in order to download and compile deal.II. Note that the library relies on [p4est](https://www.p4est.org/) in order to handle distributed meshes and you might need to adjust the `P4EST_DIR` according to your installation. If you have not yet installed p4est, you might want to download the [latest tarball](https://p4est.github.io/release/p4est-2.2.tar.gz) and run the `p4est-setup.sh` script located in your deal.II directory at `dealii/doc/external-libs`. A complete installation guide is also given on the [deal.II installation page](https://dealii.org/developer/readme.html#installation). 
```
git clone https://github.com/dealii/dealii.git
mkdir build && cd build

cmake \
    -D CMAKE_BUILD_TYPE="DebugRelease" \
    -D CMAKE_CXX_FLAGS="-march=native -std=c++17" \
    -D DEAL_II_CXX_FLAGS_RELEASE="-O3" \
    -D DEAL_II_WITH_MPI="ON" \
    -D DEAL_II_WITH_P4EST="ON" \
    -D P4EST_DIR="../doc/external-libs/p4est-install"
    -D DEAL_II_WITH_LAPACK="ON" \
    -D DEAL_II_WITH_HDF5="OFF" \
    -D DEAL_II_WITH_MUPARSER="ON" \
    -D DEAL_II_WITH_THREADS="ON" \
    -D DEAL_II_COMPONENT_EXAMPLES="OFF" \
    ../dealii

make -j8
```

### Step 2: install preCICE
The program requires at least version 2.0 or greater. A nice overview of various installation options is given on the [preCICE installation page](https://www.precice.org/installation-overview.html). 
