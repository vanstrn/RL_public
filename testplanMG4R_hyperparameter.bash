echo $1

#Entropy Test
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BETA1E2_ENTLR1E7_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.01,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BETA1E3_ENTLR1E7_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.001,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BETA1E4_ENTLR1E7_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0001,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BETA1E5_ENTLR1E7_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.00001,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BETA1E6_ENTLR1E7_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.00001,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0

python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BETA1E2_ENTLR1E6_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.01,"LR Actor":0.00005,"LR Entropy":0.000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BETA1E3_ENTLR1E6_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.001,"LR Actor":0.00005,"LR Entropy":0.000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BETA1E4_ENTLR1E6_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0001,"LR Actor":0.00005,"LR Entropy":0.000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BETA1E5_ENTLR1E6_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.00001,"LR Actor":0.00005,"LR Entropy":0.000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BETA1E6_ENTLR1E6_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.00001,"LR Actor":0.00005,"LR Entropy":0.000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0

python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BETA1E2_ENTLR1E5_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.01,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BETA1E3_ENTLR1E5_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.001,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BETA1E4_ENTLR1E5_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0001,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BETA1E5_ENTLR1E5_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.00001,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BETA1E6_ENTLR1E5_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.00001,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0



#Learning Rate Tests
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"LR5E5_CB5E1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"LR5E5_CB7E1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.7,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"LR5E5_CB1E0_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":1.0,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0

python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"LR1E5_CB5E1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00001,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"LR1E5_CB7E1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00001,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.7,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"LR1E5_CB1E0_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00001,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":1.0,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0

python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"LR1E4_CB5E1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.0001,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"LR1E4_CB7E1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.0001,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.7,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"LR1E4_CB1E0_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.0001,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":1.0,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0


#Batch Size Tests

python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BS1024_MB16_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":16,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BS1024_MB32_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BS1024_MB64_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":64,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BS1024_MB64_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":128,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0

python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BS512_MB16_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":512,"MinibatchSize":16,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BS512_MB32_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BS512_MB64_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":512,"MinibatchSize":64,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BS512_MB64_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":512,"MinibatchSize":128,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0

python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BS256_MB16_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":256,"MinibatchSize":16,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BS256_MB32_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":256,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BS256_MB64_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":256,"MinibatchSize":64,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_3.json","RunName":"BS256_MB64_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":256,"MinibatchSize":128,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0
