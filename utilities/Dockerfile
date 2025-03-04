FROM precice/precice:nightly as stage1

# This is incomplete, the precice nightly installs in /usr/, we also need to copy the cmake stuff here, but it's content would be incorrect
FROM dealii/dealii:master-noble
COPY --from=stage1 /usr/lib/x86_64-linux-gnu/libprecice* /usr/local/lib/
COPY --from=stage1 /usr/include/precice/ /usr/local/include/precice/
COPY --from=stage1 /usr/bin/precice-* /usr/local/bin/
