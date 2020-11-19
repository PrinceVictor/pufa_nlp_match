#!/usr/bin/env bash
unzip $1
if [ ! -d "data/train" ];then
    mkdir data/train
fi
if [ ! -d "data/val" ];then
    mkdir data/val
fi
python data_split.py
cp data/db_schema.json data/val/db_schema.json
mv data/db_schema.json data/train/db_schema.json
cp data/db_content.json data/val/db_content.json
mv data/db_content.json data/train/db_content.json

rm -f data/train.json
rm -rf _*
