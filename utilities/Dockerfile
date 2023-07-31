FROM precice/precice:develop

FROM dealii/dealii:master-jammy
COPY --from=0 /usr/local/lib /usr/local/lib
COPY --from=0 /usr/local/include /usr/local/include
COPY --from=0 /usr/local/bin /usr/local/bin
