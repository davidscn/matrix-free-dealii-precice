# Coupled regression tests

The regression tests are registered with CTest when `BUILD_TESTING=ON`.
Configure the project and build the application executable that should be
tested:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target solid
ctest --test-dir build -L solid --output-on-failure
```

For the heat-transfer suite, build `heat` and select the `heat` label:

```bash
cmake --build build --target heat
ctest --test-dir build -L heat --output-on-failure
```

Every physical case is tested with one solver rank (`serial`) and four solver
ranks (`parallel`). These modes are also available as CTest labels. The number
of ranks and the 120-second timeout can be configured with
`COUPLED_TEST_MPI_RANKS` and `COUPLED_TEST_TIMEOUT`, respectively.
`SOLID_TEST_CG_RELATIVE_TOLERANCE` and
`SOLID_TEST_CG_ABSOLUTE_TOLERANCE` control the accepted increase in the
average CG iteration count; the larger allowance is used and lower counts are
always accepted. Heat error values use the configurable absolute tolerance
`HEAT_TEST_ABSOLUTE_TOLERANCE`.

CTest builds the lightweight `dummy_tester` automatically before solid tests.
It does not build `solid` or `heat`. Before each coupled run, a setup fixture
removes and recreates that test's directory below `<build>/tests/work`. This
also removes previous solver output and the `precice-run` communication
directory.
