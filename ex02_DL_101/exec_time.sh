#!/bin/bash

> exec_time.txt

workers_values=(1 2 4 8 16)

for workers in $"${workers_values[@]}"
do
	echo "Worker(s): $workers" >> exec_time.txt
	{ time ./run_cifar.py --num_worker "$workers" ; } 2>> exec_time.txt
done
