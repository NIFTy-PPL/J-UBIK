#!/bin/bash

python3 pdftable/csv2tex.py
pdflatex pdftable/obs.tex
rm obs.out obs.log obs.aux
