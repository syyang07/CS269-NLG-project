#!/bin/bash
AWKCOMMAND='
NR==FNR {a[$1];next} 
!(FNR in a) {print > FILENAME".not"}
'

cat -n tokenized_context.txt | shuf -n 6000000 --random-source=tokenized_context.txt | sort -n > 6M_context.txt
cut -f2- 6M_context.txt > train_context.txt
cat -n tokenized_reviews.txt | shuf -n 6000000 --random-source=tokenized_context.txt | sort -n | cut -f2- > train_reviews.txt

awk "$AWKCOMMAND" <(cut -f1 6M_context.txt) train_context.txt train_reviews.txt

cat -n train_context.txt.not | shuf -n 15000 --random-source=tokenized_context.txt | sort -n > 15k_context.txt
cut -f2- 15k_context.txt > val_context.txt
cat -n train_reviews.txt.not | shuf -n 15000 --random-source=tokenized_context.txt | sort -n | cut -f2- > val_reviews.txt

awk "$AWKCOMMAND" <(cut -f1 15k_context.txt) val_context.txt val_reviews.txt


