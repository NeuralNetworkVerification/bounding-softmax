#!/usr/bin/env bash

for network in self-attention-sst-sim-post-embedding.onnx
do
    #for i in {0..2}
    for i in {0..499}
    do
	for t in linear er lse1 lse2
	do
	    for e in 0.015 0.0025 0.0035 0.0045 0.005 0.0055 0.006
	    do
		echo $network $i $e $t
	    done
	done
    done
done
