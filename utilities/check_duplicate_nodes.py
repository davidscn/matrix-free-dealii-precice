#! /usr/bin/env python3
# Searches in the given input file (in particular the vtk exports) for duplicated lines.
# RBF mappings will fail to converge if there are duplications in the 'from' mesh (consistent) or 'to' mesh (conservative)
# If the script returns nothing, no duplication was found.
# Axis aligned constraints should only be set if there are no additinal information (only one layer) in a respective direction.

import sys

with open(str(sys.argv[1])) as f:
     seen = set()
     for line in f:
         line_lower = line.lower()

         if "CELLS" in line:
            sys.exit()

         if line_lower in seen and line_lower.strip():
             print(line)
         else:
             seen.add(line_lower)
