#!/bin/bash

#Shell script wrapper to run all deep learning simulations.
#Author: David Nahmias


start=$SECONDS

for (( t=1; t<=10; t++ )); do
	python auto_diagnosis.py 'dilantin','keppra',0,${t}
	wait
done

for (( t=1; t<=10; t++ )); do
	python auto_diagnosis.py 'dilantin','keppra',1,${t}
	wait
done

for (( t=1; t<=10; t++ )); do
	python auto_diagnosis.py 'none','dilantin',0,${t}
	wait
done

for (( t=1; t<=10; t++ )); do
	python auto_diagnosis.py 'none','keppra',0,${t}
	wait
done

for (( t=1; t<=10; t++ )); do
	python auto_diagnosis.py 'none','dilantin',1,${t}
	wait
done

for (( t=1; t<=10; t++ )); do
	python auto_diagnosis.py 'none','keppra',1,${t}
	wait
done

echo 'Time elapsed: ' $(( SECONDS - start ))
echo 'Finished all simulation'