#!/usr/bin/env bash

for i in {0..99}
do
    for scoring in Brier NLL
    do
	for eps in 0.008 0.012 0.016
	do
	    echo $eps $index ER LSE $scoring
	done
    done
done

