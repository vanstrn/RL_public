echo $1

python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BETA1E6_ENTLR1E7_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.00001,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}'
