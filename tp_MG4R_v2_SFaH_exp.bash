echo "Batch Number" $1
if [ -z "$2" ]
  then
    echo "Device argument not supplied defaulting to CPU"
    DEVICE=cpu
  else
    DEVICE=$2
fi
echo "Using device:" $DEVICE

python SF_v2_action.py -f MG4R_v2_SFa.json -c '{"RunName":"MG4R_v2_SFa_128_'$1'"}' -n '{"SFSize":"128"}' -p /$DEVICE:0

python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_128_'$1'_1_PPO_HP","LoadName":"MG4R_v2_SFa_128_'$1'","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_128_'$1'_1_PPO_RS","LoadName":"MG4R_v2_SFa_128_'$1'","TotalSamples":128,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_128_'$1'_1_PPO_FS","LoadName":"MG4R_v2_SFa_128_'$1'","TotalSamples":128,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_128_'$1'_1_PPO_RN","LoadName":"MG4R_v2_SFa_128_'$1'","TotalSamples":128,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_128_'$1'_1_PPO_HC","LoadName":"MG4R_v2_SFa_128_'$1'","TotalSamples":128,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait

python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_128_'$1'_2_PPO_HP","LoadName":"MG4R_v2_SFa_128_'$1'","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_128_'$1'_2_PPO_RS","LoadName":"MG4R_v2_SFa_128_'$1'","TotalSamples":128,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_128_'$1'_2_PPO_FS","LoadName":"MG4R_v2_SFa_128_'$1'","TotalSamples":128,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_128_'$1'_2_PPO_RN","LoadName":"MG4R_v2_SFa_128_'$1'","TotalSamples":128,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_128_'$1'_2_PPO_HC","LoadName":"MG4R_v2_SFa_128_'$1'","TotalSamples":128,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait

python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_128_'$1'_3_PPO_HP","LoadName":"MG4R_v2_SFa_128_'$1'","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_128_'$1'_3_PPO_RS","LoadName":"MG4R_v2_SFa_128_'$1'","TotalSamples":128,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_128_'$1'_3_PPO_FS","LoadName":"MG4R_v2_SFa_128_'$1'","TotalSamples":128,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_128_'$1'_3_PPO_RN","LoadName":"MG4R_v2_SFa_128_'$1'","TotalSamples":128,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_128_'$1'_3_PPO_HC","LoadName":"MG4R_v2_SFa_128_'$1'","TotalSamples":128,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait


python SF_v2_action.py -f MG4R_v2_SFa.json -c '{"RunName":"MG4R_v2_SFa_96_'$1'"}' -n '{"SFSize":"96"}' -p /$DEVICE:0

python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_96_'$1'_1_PPO_HP","LoadName":"MG4R_v2_SFa_96_'$1'","TotalSamples":96,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_96_'$1'_1_PPO_RS","LoadName":"MG4R_v2_SFa_96_'$1'","TotalSamples":96,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_96_'$1'_1_PPO_FS","LoadName":"MG4R_v2_SFa_96_'$1'","TotalSamples":96,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_96_'$1'_1_PPO_RN","LoadName":"MG4R_v2_SFa_96_'$1'","TotalSamples":96,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_96_'$1'_1_PPO_HC","LoadName":"MG4R_v2_SFa_96_'$1'","TotalSamples":96,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait

python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_96_'$1'_2_PPO_HP","LoadName":"MG4R_v2_SFa_96_'$1'","TotalSamples":96,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_96_'$1'_2_PPO_RS","LoadName":"MG4R_v2_SFa_96_'$1'","TotalSamples":96,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_96_'$1'_2_PPO_FS","LoadName":"MG4R_v2_SFa_96_'$1'","TotalSamples":96,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_96_'$1'_2_PPO_RN","LoadName":"MG4R_v2_SFa_96_'$1'","TotalSamples":96,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_96_'$1'_2_PPO_HC","LoadName":"MG4R_v2_SFa_96_'$1'","TotalSamples":96,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait

python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_96_'$1'_3_PPO_HP","LoadName":"MG4R_v2_SFa_96_'$1'","TotalSamples":96,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_96_'$1'_3_PPO_RS","LoadName":"MG4R_v2_SFa_96_'$1'","TotalSamples":96,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_96_'$1'_3_PPO_FS","LoadName":"MG4R_v2_SFa_96_'$1'","TotalSamples":96,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_96_'$1'_3_PPO_RN","LoadName":"MG4R_v2_SFa_96_'$1'","TotalSamples":96,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_96_'$1'_3_PPO_HC","LoadName":"MG4R_v2_SFa_96_'$1'","TotalSamples":96,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait


python SF_v2_action.py -f MG4R_v2_SFa.json -c '{"RunName":"MG4R_v2_SFa_64_'$1'"}' -n '{"SFSize":"64"}' -p /$DEVICE:0

python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_64_'$1'_1_PPO_HP","LoadName":"MG4R_v2_SFa_64_'$1'","TotalSamples":64,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_64_'$1'_1_PPO_RS","LoadName":"MG4R_v2_SFa_64_'$1'","TotalSamples":64,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_64_'$1'_1_PPO_FS","LoadName":"MG4R_v2_SFa_64_'$1'","TotalSamples":64,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_64_'$1'_1_PPO_RN","LoadName":"MG4R_v2_SFa_64_'$1'","TotalSamples":64,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_64_'$1'_1_PPO_HC","LoadName":"MG4R_v2_SFa_64_'$1'","TotalSamples":64,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait

python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_64_'$1'_2_PPO_HP","LoadName":"MG4R_v2_SFa_64_'$1'","TotalSamples":64,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_64_'$1'_2_PPO_RS","LoadName":"MG4R_v2_SFa_64_'$1'","TotalSamples":64,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_64_'$1'_2_PPO_FS","LoadName":"MG4R_v2_SFa_64_'$1'","TotalSamples":64,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_64_'$1'_2_PPO_RN","LoadName":"MG4R_v2_SFa_64_'$1'","TotalSamples":64,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_64_'$1'_2_PPO_HC","LoadName":"MG4R_v2_SFa_64_'$1'","TotalSamples":64,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait

python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_64_'$1'_3_PPO_HP","LoadName":"MG4R_v2_SFa_64_'$1'","TotalSamples":64,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_64_'$1'_3_PPO_RS","LoadName":"MG4R_v2_SFa_64_'$1'","TotalSamples":64,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_64_'$1'_3_PPO_FS","LoadName":"MG4R_v2_SFa_64_'$1'","TotalSamples":64,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_64_'$1'_3_PPO_RN","LoadName":"MG4R_v2_SFa_64_'$1'","TotalSamples":64,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_64_'$1'_3_PPO_HC","LoadName":"MG4R_v2_SFa_64_'$1'","TotalSamples":64,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait


python SF_v2_action.py -f MG4R_v2_SFa.json -c '{"RunName":"MG4R_v2_SFa_48_'$1'"}' -n '{"SFSize":"48"}' -p /$DEVICE:0

python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_48_'$1'_1_PPO_HP","LoadName":"MG4R_v2_SFa_48_'$1'","TotalSamples":48,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_48_'$1'_1_PPO_RS","LoadName":"MG4R_v2_SFa_48_'$1'","TotalSamples":48,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_48_'$1'_1_PPO_FS","LoadName":"MG4R_v2_SFa_48_'$1'","TotalSamples":48,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_48_'$1'_1_PPO_RN","LoadName":"MG4R_v2_SFa_48_'$1'","TotalSamples":48,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_48_'$1'_1_PPO_HC","LoadName":"MG4R_v2_SFa_48_'$1'","TotalSamples":48,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait

python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_48_'$1'_2_PPO_HP","LoadName":"MG4R_v2_SFa_48_'$1'","TotalSamples":48,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_48_'$1'_2_PPO_RS","LoadName":"MG4R_v2_SFa_48_'$1'","TotalSamples":48,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_48_'$1'_2_PPO_FS","LoadName":"MG4R_v2_SFa_48_'$1'","TotalSamples":48,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_48_'$1'_2_PPO_RN","LoadName":"MG4R_v2_SFa_48_'$1'","TotalSamples":48,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_48_'$1'_2_PPO_HC","LoadName":"MG4R_v2_SFa_48_'$1'","TotalSamples":48,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait

python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_48_'$1'_3_PPO_HP","LoadName":"MG4R_v2_SFa_48_'$1'","TotalSamples":48,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_48_'$1'_3_PPO_RS","LoadName":"MG4R_v2_SFa_48_'$1'","TotalSamples":48,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_48_'$1'_3_PPO_FS","LoadName":"MG4R_v2_SFa_48_'$1'","TotalSamples":48,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_48_'$1'_3_PPO_RN","LoadName":"MG4R_v2_SFa_48_'$1'","TotalSamples":48,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFaH_48_'$1'_3_PPO_HC","LoadName":"MG4R_v2_SFa_48_'$1'","TotalSamples":48,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait
