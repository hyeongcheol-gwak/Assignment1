#!/bin/bash

usage(){
    echo "Usage: $0 student-number"
    exit 1
}

[[ $# -eq 0 ]] && usage

tar czvf ${1}.tar.gz AS1-Logistic_Regression.ipynb AS1-Logistic_Regression.py AS1-Logistic_Regression.md
