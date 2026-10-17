#!/bin/sh
# Doxygen input filter: the headers use plain /* ... */ comments for
# documentation; turn them into Doxygen comments (/** ... */) on the fly.
sed -e 's|^\( *\)/\* |\1/** |' -e 's|^\( *\)/\*  |\1/** |' "$1"
