#!/bin/bash
python $2  --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Real_obda.txt --target ./txt/target_Clipart_obda.txt --gpu $1 --exp_name R2C_ent
python $2  --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Real_obda.txt --target ./txt/target_Art_obda.txt --gpu $1 --exp_name R2A_ent
python $2  --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Real_obda.txt --target ./txt/target_Product_obda.txt --gpu $1 --exp_name R2P_ent
python $2  --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Product_obda.txt --target ./txt/target_Art_obda.txt --gpu $1 --exp_name P2A_ent
python $2  --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Product_obda.txt --target ./txt/target_Clipart_obda.txt --gpu $1 --exp_name P2C_ent
python $2  --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Product_obda.txt --target ./txt/target_Real_obda.txt --gpu $1 --exp_name P2R_ent
python $2  --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Clipart_obda.txt --target ./txt/target_Real_obda.txt --gpu $1 --exp_name C2R_ent
python $2  --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Clipart_obda.txt --target ./txt/target_Art_obda.txt --gpu $1 --exp_name C2A_ent
python $2  --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Clipart_obda.txt --target ./txt/target_Product_obda.txt --gpu $1 --exp_name C2P_ent
python $2  --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Art_obda.txt --target ./txt/target_Product_obda.txt --gpu $1 --exp_name A2P_ent
python $2  --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Art_obda.txt --target ./txt/target_Real_obda.txt --gpu $1 --exp_name A2R_ent
python $2  --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Art_obda.txt --target ./txt/target_Clipart_obda.txt --gpu $1 --exp_name A2C_ent
