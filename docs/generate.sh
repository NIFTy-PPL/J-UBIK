#!/usr/bin/sh

set -e

FOLDER=docs/source/user

# for FILE in ${FOLDER}/0_intro ${FOLDER}/old_nifty_getting_started_0 ${FOLDER}/old_nifty_getting_started_4_CorrelatedFields ${FOLDER}/old_nifty_custom_nonlinearities ${FOLDER}/niftyre_getting_started_4_CorrelatedFields; do
#     if [ ! -f "${FILE}.md" ] || [ "${FILE}.ipynb" -nt "${FILE}.md" ]; then
# 		jupytext --to ipynb "${FILE}.py"
#         jupyter-nbconvert --to markdown --execute --ExecutePreprocessor.timeout=None "${FILE}.ipynb"
#     fi
# done

EXCLUDE=""
sphinx-apidoc -e -o  docs/source/mod jubik0 ${EXCLUDE}
sphinx-build -b html docs/source/ docs/build/