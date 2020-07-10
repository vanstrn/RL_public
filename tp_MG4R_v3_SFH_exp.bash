echo "Batch Number" $1
if [ -z "$2" ]
  then
    echo "Device argument not supplied defaulting to CPU"
    DEVICE=cpu
  else
    DEVICE=$2
fi
echo "Using device:" $DEVICE

python SF_v2.py -f MG4R_v3_SF.json -c '{"RunName":"MG4R_v3_SF_512_'$1'"}' -n '{"SFSize":"512"}' -p /$DEVICE:0

python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_512_'$1'_1_PPO_HP","LoadName":"MG4R_v3_SF_512_'$1'","TotalSamples":512,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_512_'$1'_1_PPO_RS","LoadName":"MG4R_v3_SF_512_'$1'","TotalSamples":512,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_512_'$1'_1_PPO_FS","LoadName":"MG4R_v3_SF_512_'$1'","TotalSamples":512,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_512_'$1'_1_PPO_RN","LoadName":"MG4R_v3_SF_512_'$1'","TotalSamples":512,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_512_'$1'_1_PPO_HC","LoadName":"MG4R_v3_SF_512_'$1'","TotalSamples":512,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &

wait

python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_512_'$1'_2_PPO_HP","LoadName":"MG4R_v3_SF_512_'$1'","TotalSamples":512,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_512_'$1'_2_PPO_RS","LoadName":"MG4R_v3_SF_512_'$1'","TotalSamples":512,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_512_'$1'_2_PPO_FS","LoadName":"MG4R_v3_SF_512_'$1'","TotalSamples":512,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_512_'$1'_2_PPO_RN","LoadName":"MG4R_v3_SF_512_'$1'","TotalSamples":512,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_512_'$1'_2_PPO_HC","LoadName":"MG4R_v3_SF_512_'$1'","TotalSamples":512,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
wait

python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_512_'$1'_3_PPO_HP","LoadName":"MG4R_v3_SF_512_'$1'","TotalSamples":512,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_512_'$1'_3_PPO_RS","LoadName":"MG4R_v3_SF_512_'$1'","TotalSamples":512,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_512_'$1'_3_PPO_FS","LoadName":"MG4R_v3_SF_512_'$1'","TotalSamples":512,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_512_'$1'_3_PPO_RN","LoadName":"MG4R_v3_SF_512_'$1'","TotalSamples":512,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_512_'$1'_3_PPO_HC","LoadName":"MG4R_v3_SF_512_'$1'","TotalSamples":512,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
wait


python SF_v2.py -f MG4R_v3_SF.json -c '{"RunName":"MG4R_v3_SF_256_'$1'"}' -n '{"SFSize":"256"}' -p /$DEVICE:0

python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_256_'$1'_1_PPO_HP","LoadName":"MG4R_v3_SF_256_'$1'","TotalSamples":256,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_256_'$1'_1_PPO_RS","LoadName":"MG4R_v3_SF_256_'$1'","TotalSamples":256,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_256_'$1'_1_PPO_FS","LoadName":"MG4R_v3_SF_256_'$1'","TotalSamples":256,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_256_'$1'_1_PPO_RN","LoadName":"MG4R_v3_SF_256_'$1'","TotalSamples":256,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_256_'$1'_1_PPO_HC","LoadName":"MG4R_v3_SF_256_'$1'","TotalSamples":256,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
wait

python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_256_'$1'_2_PPO_HP","LoadName":"MG4R_v3_SF_256_'$1'","TotalSamples":256,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_256_'$1'_2_PPO_RS","LoadName":"MG4R_v3_SF_256_'$1'","TotalSamples":256,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_256_'$1'_2_PPO_FS","LoadName":"MG4R_v3_SF_256_'$1'","TotalSamples":256,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_256_'$1'_2_PPO_RN","LoadName":"MG4R_v3_SF_256_'$1'","TotalSamples":256,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_256_'$1'_2_PPO_HC","LoadName":"MG4R_v3_SF_256_'$1'","TotalSamples":256,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
wait

python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_256_'$1'_3_PPO_HP","LoadName":"MG4R_v3_SF_256_'$1'","TotalSamples":256,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_256_'$1'_3_PPO_RS","LoadName":"MG4R_v3_SF_256_'$1'","TotalSamples":256,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_256_'$1'_3_PPO_FS","LoadName":"MG4R_v3_SF_256_'$1'","TotalSamples":256,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_256_'$1'_3_PPO_RN","LoadName":"MG4R_v3_SF_256_'$1'","TotalSamples":256,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_256_'$1'_3_PPO_HC","LoadName":"MG4R_v3_SF_256_'$1'","TotalSamples":256,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
wait


python SF_v2.py -f MG4R_v3_SF.json -c '{"RunName":"MG4R_v3_SF_128_'$1'"}' -n '{"SFSize":"128"}' -p /$DEVICE:0

python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_128_'$1'_1_PPO_HP","LoadName":"MG4R_v3_SF_128_'$1'","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_128_'$1'_1_PPO_RS","LoadName":"MG4R_v3_SF_128_'$1'","TotalSamples":128,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_128_'$1'_1_PPO_FS","LoadName":"MG4R_v3_SF_128_'$1'","TotalSamples":128,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_128_'$1'_1_PPO_RN","LoadName":"MG4R_v3_SF_128_'$1'","TotalSamples":128,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_128_'$1'_1_PPO_HC","LoadName":"MG4R_v3_SF_128_'$1'","TotalSamples":128,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
wait

python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_128_'$1'_2_PPO_HP","LoadName":"MG4R_v3_SF_128_'$1'","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_128_'$1'_2_PPO_RS","LoadName":"MG4R_v3_SF_128_'$1'","TotalSamples":128,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_128_'$1'_2_PPO_FS","LoadName":"MG4R_v3_SF_128_'$1'","TotalSamples":128,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_128_'$1'_2_PPO_RN","LoadName":"MG4R_v3_SF_128_'$1'","TotalSamples":128,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_128_'$1'_2_PPO_HC","LoadName":"MG4R_v3_SF_128_'$1'","TotalSamples":128,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
wait

python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_128_'$1'_3_PPO_HP","LoadName":"MG4R_v3_SF_128_'$1'","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_128_'$1'_3_PPO_RS","LoadName":"MG4R_v3_SF_128_'$1'","TotalSamples":128,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_128_'$1'_3_PPO_FS","LoadName":"MG4R_v3_SF_128_'$1'","TotalSamples":128,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_128_'$1'_3_PPO_RN","LoadName":"MG4R_v3_SF_128_'$1'","TotalSamples":128,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_128_'$1'_3_PPO_HC","LoadName":"MG4R_v3_SF_128_'$1'","TotalSamples":128,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
wait


python SF_v2.py -f MG4R_v3_SF.json -c '{"RunName":"MG4R_v3_SF_64_'$1'"}' -n '{"SFSize":"64"}' -p /$DEVICE:0

python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_64_'$1'_1_PPO_HP","LoadName":"MG4R_v3_SF_64_'$1'","TotalSamples":64,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_64_'$1'_1_PPO_RS","LoadName":"MG4R_v3_SF_64_'$1'","TotalSamples":64,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_64_'$1'_1_PPO_FS","LoadName":"MG4R_v3_SF_64_'$1'","TotalSamples":64,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_64_'$1'_1_PPO_RN","LoadName":"MG4R_v3_SF_64_'$1'","TotalSamples":64,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_64_'$1'_1_PPO_HC","LoadName":"MG4R_v3_SF_64_'$1'","TotalSamples":64,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
wait

python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_64_'$1'_2_PPO_HP","LoadName":"MG4R_v3_SF_64_'$1'","TotalSamples":64,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_64_'$1'_2_PPO_RS","LoadName":"MG4R_v3_SF_64_'$1'","TotalSamples":64,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_64_'$1'_2_PPO_FS","LoadName":"MG4R_v3_SF_64_'$1'","TotalSamples":64,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_64_'$1'_2_PPO_RN","LoadName":"MG4R_v3_SF_64_'$1'","TotalSamples":64,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_64_'$1'_2_PPO_HC","LoadName":"MG4R_v3_SF_64_'$1'","TotalSamples":64,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
wait

python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_64_'$1'_3_PPO_HP","LoadName":"MG4R_v3_SF_64_'$1'","TotalSamples":64,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_64_'$1'_3_PPO_RS","LoadName":"MG4R_v3_SF_64_'$1'","TotalSamples":64,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_64_'$1'_3_PPO_FS","LoadName":"MG4R_v3_SF_64_'$1'","TotalSamples":64,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_64_'$1'_3_PPO_RN","LoadName":"MG4R_v3_SF_64_'$1'","TotalSamples":64,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v3_SFH.json -c '{"RunName":"MG4R_v3_SFH_64_'$1'_3_PPO_HC","LoadName":"MG4R_v3_SF_64_'$1'","TotalSamples":64,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
wait