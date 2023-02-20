#!/usr/bin/env bash

for network in self-attention-sst-sim-post-embedding.onnx
do
    #for i in {0..2}
    for i in {0..499}
    do
	for t in linear er lse1 lse2
	do
	    for e in 0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16
	    do
		echo $network $i $e $t
	    done
	done
    done
done
