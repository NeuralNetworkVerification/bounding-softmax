#!/usr/bin/env bash

for i in {0..99}
do
    for scoring in Brier NLL
    do
	for eps in 0.008 0.012 0.016
	do
	    echo $eps $i ER LSE $scoring
	    echo $eps $i LSE LSE $scoring
	    echo $eps $i lin lin $scoring
	done
    done
done

