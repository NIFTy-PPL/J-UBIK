#!/bin/bash

python3 pdftable/csv2tex.py
sed -i 's/tabular/supertabular/g' pdftable/tab.tex
pdflatex pdftable/obs.tex
rm obs.out obs.log obs.aux
