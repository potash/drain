#!/bin/bash
# dump grid_search.py output into a temporary drakefile and run it

basename=$(dirname $0)
TMPFILE=`mktemp` || exit 1
python $basename/grid_search.py "$@" > $TMPFILE && drake -w $TMPFILE
