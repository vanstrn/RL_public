echo $1

python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_1.json","RunName":"MG4R_v2_1_PPO_v5_net_v1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json","NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005}}'
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"MG4R_v2_3_PPO_v5_net_v1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json","NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005}}'
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"MG4R_v2_4_PPO_v5_net_v1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json","NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005}}'
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_2.json","RunName":"MG4R_v2_2_PPO_v5_net_v1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json","NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005}}'

python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_1.json","RunName":"MG4R_v2_1_PPO_v5_net_v1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json","NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.0001}}'
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"MG4R_v2_3_PPO_v5_net_v1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json","NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.0001}}'
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"MG4R_v2_4_PPO_v5_net_v1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json","NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.0001}}'
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_2.json","RunName":"MG4R_v2_2_PPO_v5_net_v1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json","NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.0001}}'

python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_1.json","RunName":"MG4R_v2_1_PPO_v5_net_v1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json","NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00001}}'
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"MG4R_v2_3_PPO_v5_net_v1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json","NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00001}}'
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"MG4R_v2_4_PPO_v5_net_v1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json","NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00001}}'
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_2.json","RunName":"MG4R_v2_2_PPO_v5_net_v1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json","NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00001}}'
