#!/bin/bash

# generates a coverage badge based on the total line coverage output
# takes 2 arguments:
#     ${1} filename of coverage output
#     ${2} filename of the badge

total=$(grep 'Total coverage:' ${1} | grep -Po '[0-9.]+(?=%$)')
echo "Generating badge for Total Coverage of $total"

if (( $(echo "$total <= 60" | bc -l) )) ; then
    COLOR=red
elif (( $(echo "$total > 90" | bc -l) )); then
    COLOR=green
else
    COLOR=orange
fi

curl "https://img.shields.io/badge/coverage-$total%25-$COLOR" > ${2}
