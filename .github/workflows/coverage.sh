#!/bin/bash

# generates a coverage badge based on the total line coverage output
# takes 2 arguments:
#     ${1} filename of coverage output
#     ${2} filename of the badge

total=$(grep 'TOTAL' ${1} | grep -Po '[0-9]+(?=%$)')
echo "Generating badge for Total Coverage of $total"

if (( $(echo "$total < 70" | bc -l) )) ; then
    COLOR=red
elif (( $(echo "$total >= 90" | bc -l) )); then
    COLOR=green
else
    COLOR=orange
fi

mkdir -p $(dirname ${2})
echo "generating coverage badge in $(dirname ${2})"
curl "https://img.shields.io/badge/coverage-$total%25-$COLOR" > ${2}
