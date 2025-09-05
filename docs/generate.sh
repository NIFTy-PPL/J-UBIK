#!/usr/bin/sh

set -e

FOLDER=docs/source/user

# execute these
for FILE in ${FOLDER}/spectral_sky_demo; do
    if [ ! -f "${FILE}.md" ] || [ "${FILE}.ipynb" -nt "${FILE}.md" ]; then
		jupytext --to ipynb "${FILE}.py"
        jupyter-nbconvert --to markdown --execute --ExecutePreprocessor.timeout=None "${FILE}.ipynb"
    fi
done

# don't execute these
for FILE in ${FOLDER}/chandra_likelihood_demo; do
    if [ ! -f "${FILE}.md" ] || [ "${FILE}.ipynb" -nt "${FILE}.md" ]; then
		jupytext --to ipynb "${FILE}.py"
        jupyter-nbconvert --to markdown "${FILE}.ipynb"
    fi
done

EXCLUDE=""
sphinx-apidoc -e -d 1 -o  docs/source/mod jubik0 ${EXCLUDE}
sphinx-build -b html docs/source/ docs/build/
