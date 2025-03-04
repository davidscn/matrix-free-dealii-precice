FROM precice/precice:nightly as stage1

FROM dealii/dealii:master-noble
USER root
COPY --from=stage1 /usr/local/lib /usr/local/lib
COPY --from=stage1 /usr/local/include /usr/local/include
COPY --from=stage1 /usr/local/bin /usr/local/bin

RUN ldconfig

USER dealii
