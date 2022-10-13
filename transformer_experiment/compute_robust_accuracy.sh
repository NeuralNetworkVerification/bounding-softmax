#!/usr/bin/env bash

for network in small-pgd-sim.onnx med-pgd-sim.onnx big-pgd-sim.onnx
do
    for e in  0.016 0.02 0.024 0.032
    do
	for t in lin0 linear er lse1 lse2 lse0.6 lse0.8
	do
	    numunsat=$(grep "$network" results.csv | grep "$e","$t",unsat | wc -l)
	    echo "$network","$e","$t",$numunsat
	done
    done
done
