echo $1

#Entropy Test
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_BETA1E3_ENTLR1E7_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.001,"LR Actor":0.00001,"LR Entropy":0.0000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_BETA1E4_ENTLR1E7_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0001,"LR Actor":0.00001,"LR Entropy":0.0000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_BETA1E5_ENTLR1E7_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.00001,"LR Actor":0.00001,"LR Entropy":0.0000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_BETA1E6_ENTLR1E7_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.000001,"LR Actor":0.00001,"LR Entropy":0.0000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
wait

python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_BETA1E3_ENTLR1E6_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.001,"LR Actor":0.00001,"LR Entropy":0.000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_BETA1E4_ENTLR1E6_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0001,"LR Actor":0.00001,"LR Entropy":0.000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_BETA1E5_ENTLR1E6_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.00001,"LR Actor":0.00001,"LR Entropy":0.000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_BETA1E6_ENTLR1E6_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.000001,"LR Actor":0.00001,"LR Entropy":0.000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
wait

python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_BETA1E3_ENTLR1E5_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.001,"LR Actor":0.00001,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_BETA1E4_ENTLR1E5_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0001,"LR Actor":0.00001,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_BETA1E5_ENTLR1E5_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.00001,"LR Actor":0.00001,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_BETA1E6_ENTLR1E5_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.000001,"LR Actor":0.00001,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
wait


#Learning Rate Tests
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_LR5E5_CB5E1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_LR5E5_CB7E1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.7,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_LR5E5_CB1E0_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":1.0,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_LR5E5_CB2E0_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":2.0,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
wait

python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_LR1E5_CB5E1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00001,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_LR1E5_CB7E1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00001,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.7,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_LR1E5_CB1E0_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00001,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":1.0,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_LR1E5_CB2E0_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00001,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":2.0,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
wait

python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_LR1E4_CB5E1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.0001,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_LR1E4_CB7E1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.0001,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.7,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_LR1E4_CB1E0_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.0001,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":1.0,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_LR1E4_CB2E0_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.0001,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":2.0,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
wait

python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_LR5E6_CB5E1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.000005,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_LR5E6_CB7E1_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.000005,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.7,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_LR5E6_CB1E0_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.000005,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":1.0,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_LR5E6_CB2E0_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.000005,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":2.0,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
wait

#Batch Size Tests

python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_BS1024_MB16_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":16,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_BS1024_MB32_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_BS1024_MB64_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":64,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_BS1024_MB64_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":1024,"MinibatchSize":128,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
wait

python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_BS512_MB16_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":512,"MinibatchSize":16,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_BS512_MB32_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_BS512_MB64_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":512,"MinibatchSize":64,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_BS512_MB64_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":512,"MinibatchSize":128,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
wait

python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_BS256_MB16_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":256,"MinibatchSize":16,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_BS256_MB32_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":256,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_BS256_MB64_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":256,"MinibatchSize":64,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
python SingleExec_v4.py -f MG4R_v2_PPO_v2.json -c '{"EnvConfig":"MG4R_v2_4.json","RunName":"2_4_BS256_MB64_'$1'","NetworkConfig":"MG4R_v2_AC_v1.json",
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.00001,"BatchSize":256,"MinibatchSize":128,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /cpu:0 &
wait
