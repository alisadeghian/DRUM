#!/usr/bin/env bash


usage()
{
    echo "usage: sysinfo_page [[[-d dataset ] [-i]] | [-h]]"
}

dataset=""
topk=(1 3 10)

while [ "$1" != "" ]; do
	case $1 in
		-d | --dataset )        shift
                                dataset="$1"
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
	esac
	shift                                
done

echo "evaluating $dataset data set model results"

. eval/collect_all_facts.sh datasets/$dataset
python2 eval/get_truths.py datasets/$dataset

echo "Top K results" > exps/"$dataset"/topk

for i in "${topk[@]}"
do
	echo "Top $i results: " >> exps/"$dataset"/topk
	python2 eval/evaluate.py --preds=exps/"$dataset"/test_predictions.txt --truths=datasets/"$dataset"/truths.pckl --top_k=$i >> exps/"$dataset"/topk
	python2 eval/evaluate.py --preds=exps/"$dataset"/test_predictions.txt --truths=datasets/"$dataset"/truths.pckl --top_k=$i --raw >> exps/"$dataset"/topk_raw
done






