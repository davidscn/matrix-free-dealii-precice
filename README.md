# A matrix-free high performance solid solver for coupled fluid-structure interactions
![Building](https://github.com/DavidSCN/matrix-free-dealii-precice/workflows/Building/badge.svg)

## Getting started

1. Install deal.II 

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
    -D DEAL_II_FORCE_BUNDLED_BOOST="OFF" \
    -D DEAL_II_WITH_TRILINOS="OFF" \
    -D DEAL_II_WITH_THREADS="ON" \
    -D DEAL_II_COMPONENT_EXAMPLES="OFF" \
    ../dealii

make -j8
```
