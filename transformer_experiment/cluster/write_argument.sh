#!/usr/bin/env bash

for network in small-pgd-sim.onnx med-pgd-sim.onnx big-pgd-sim.onnx
do
    #for i in {0..2}
    for i in {0..499}
    do
	for t in linear lse1 lse2 lse0.6 lse0.8 er lin0
	do
	    for e in  0.016 0.02 0.024
	    do
		echo $network $i $e $t
	    done
	done
    done
done
