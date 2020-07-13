echo "Batch Number" $1
if [ -z "$2" ]
  then
    echo "Device argument not supplied defaulting to CPU"
    DEVICE=cpu
  else
    DEVICE=$2
fi
echo "Using device:" $DEVICE
if [ -z "$3" ]
  then
    echo "Environment not specified. Defaulting to v2_3"
    ENVIRON=3
  else
    ENVIRON=$3
fi
echo "Performing Experiment in environment:" $ENVIRON


python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_128_'$1'_1_'$4'_HP","LoadName":"MG4R_v2_'$ENVIRON'_SF_128_'$1'","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_128_'$1'_1_'$4'_RS","LoadName":"MG4R_v2_'$ENVIRON'_SF_128_'$1'","TotalSamples":128,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_128_'$1'_1_'$4'_FS","LoadName":"MG4R_v2_'$ENVIRON'_SF_128_'$1'","TotalSamples":128,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_128_'$1'_1_'$4'_RN","LoadName":"MG4R_v2_'$ENVIRON'_SF_128_'$1'","TotalSamples":128,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_128_'$1'_1_'$4'_HC","LoadName":"MG4R_v2_'$ENVIRON'_SF_128_'$1'","TotalSamples":128,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait

python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_128_'$1'_2_'$4'_HP","LoadName":"MG4R_v2_'$ENVIRON'_SF_128_'$1'","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_128_'$1'_2_'$4'_RS","LoadName":"MG4R_v2_'$ENVIRON'_SF_128_'$1'","TotalSamples":128,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_128_'$1'_2_'$4'_FS","LoadName":"MG4R_v2_'$ENVIRON'_SF_128_'$1'","TotalSamples":128,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_128_'$1'_2_'$4'_RN","LoadName":"MG4R_v2_'$ENVIRON'_SF_128_'$1'","TotalSamples":128,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_128_'$1'_2_'$4'_HC","LoadName":"MG4R_v2_'$ENVIRON'_SF_128_'$1'","TotalSamples":128,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait

python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_128_'$1'_3_'$4'_HP","LoadName":"MG4R_v2_'$ENVIRON'_SF_128_'$1'","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_128_'$1'_3_'$4'_RS","LoadName":"MG4R_v2_'$ENVIRON'_SF_128_'$1'","TotalSamples":128,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_128_'$1'_3_'$4'_FS","LoadName":"MG4R_v2_'$ENVIRON'_SF_128_'$1'","TotalSamples":128,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_128_'$1'_3_'$4'_RN","LoadName":"MG4R_v2_'$ENVIRON'_SF_128_'$1'","TotalSamples":128,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_128_'$1'_3_'$4'_HC","LoadName":"MG4R_v2_'$ENVIRON'_SF_128_'$1'","TotalSamples":128,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait


python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_96_'$1'_1_'$4'_HP","LoadName":"MG4R_v2_'$ENVIRON'_SF_96_'$1'","TotalSamples":96,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_96_'$1'_1_'$4'_RS","LoadName":"MG4R_v2_'$ENVIRON'_SF_96_'$1'","TotalSamples":96,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_96_'$1'_1_'$4'_FS","LoadName":"MG4R_v2_'$ENVIRON'_SF_96_'$1'","TotalSamples":96,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_96_'$1'_1_'$4'_RN","LoadName":"MG4R_v2_'$ENVIRON'_SF_96_'$1'","TotalSamples":96,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_96_'$1'_1_'$4'_HC","LoadName":"MG4R_v2_'$ENVIRON'_SF_96_'$1'","TotalSamples":96,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait

python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_96_'$1'_2_'$4'_HP","LoadName":"MG4R_v2_'$ENVIRON'_SF_96_'$1'","TotalSamples":96,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_96_'$1'_2_'$4'_RS","LoadName":"MG4R_v2_'$ENVIRON'_SF_96_'$1'","TotalSamples":96,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_96_'$1'_2_'$4'_FS","LoadName":"MG4R_v2_'$ENVIRON'_SF_96_'$1'","TotalSamples":96,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_96_'$1'_2_'$4'_RN","LoadName":"MG4R_v2_'$ENVIRON'_SF_96_'$1'","TotalSamples":96,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_96_'$1'_2_'$4'_HC","LoadName":"MG4R_v2_'$ENVIRON'_SF_96_'$1'","TotalSamples":96,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait

python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_96_'$1'_3_'$4'_HP","LoadName":"MG4R_v2_'$ENVIRON'_SF_96_'$1'","TotalSamples":96,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_96_'$1'_3_'$4'_RS","LoadName":"MG4R_v2_'$ENVIRON'_SF_96_'$1'","TotalSamples":96,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_96_'$1'_3_'$4'_FS","LoadName":"MG4R_v2_'$ENVIRON'_SF_96_'$1'","TotalSamples":96,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_96_'$1'_3_'$4'_RN","LoadName":"MG4R_v2_'$ENVIRON'_SF_96_'$1'","TotalSamples":96,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_96_'$1'_3_'$4'_HC","LoadName":"MG4R_v2_'$ENVIRON'_SF_96_'$1'","TotalSamples":96,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait


python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_64_'$1'_1_'$4'_HP","LoadName":"MG4R_v2_'$ENVIRON'_SF_64_'$1'","TotalSamples":64,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_64_'$1'_1_'$4'_RS","LoadName":"MG4R_v2_'$ENVIRON'_SF_64_'$1'","TotalSamples":64,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_64_'$1'_1_'$4'_FS","LoadName":"MG4R_v2_'$ENVIRON'_SF_64_'$1'","TotalSamples":64,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_64_'$1'_1_'$4'_RN","LoadName":"MG4R_v2_'$ENVIRON'_SF_64_'$1'","TotalSamples":64,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_64_'$1'_1_'$4'_HC","LoadName":"MG4R_v2_'$ENVIRON'_SF_64_'$1'","TotalSamples":64,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait

python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_64_'$1'_2_'$4'_HP","LoadName":"MG4R_v2_'$ENVIRON'_SF_64_'$1'","TotalSamples":64,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_64_'$1'_2_'$4'_RS","LoadName":"MG4R_v2_'$ENVIRON'_SF_64_'$1'","TotalSamples":64,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_64_'$1'_2_'$4'_FS","LoadName":"MG4R_v2_'$ENVIRON'_SF_64_'$1'","TotalSamples":64,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_64_'$1'_2_'$4'_RN","LoadName":"MG4R_v2_'$ENVIRON'_SF_64_'$1'","TotalSamples":64,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_64_'$1'_2_'$4'_HC","LoadName":"MG4R_v2_'$ENVIRON'_SF_64_'$1'","TotalSamples":64,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait

python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_64_'$1'_3_'$4'_HP","LoadName":"MG4R_v2_'$ENVIRON'_SF_64_'$1'","TotalSamples":64,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_64_'$1'_3_'$4'_RS","LoadName":"MG4R_v2_'$ENVIRON'_SF_64_'$1'","TotalSamples":64,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_64_'$1'_3_'$4'_FS","LoadName":"MG4R_v2_'$ENVIRON'_SF_64_'$1'","TotalSamples":64,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_64_'$1'_3_'$4'_RN","LoadName":"MG4R_v2_'$ENVIRON'_SF_64_'$1'","TotalSamples":64,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_64_'$1'_3_'$4'_HC","LoadName":"MG4R_v2_'$ENVIRON'_SF_64_'$1'","TotalSamples":64,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait


python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_48_'$1'_1_'$4'_HP","LoadName":"MG4R_v2_'$ENVIRON'_SF_48_'$1'","TotalSamples":48,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_48_'$1'_1_'$4'_RS","LoadName":"MG4R_v2_'$ENVIRON'_SF_48_'$1'","TotalSamples":48,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_48_'$1'_1_'$4'_FS","LoadName":"MG4R_v2_'$ENVIRON'_SF_48_'$1'","TotalSamples":48,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_48_'$1'_1_'$4'_RN","LoadName":"MG4R_v2_'$ENVIRON'_SF_48_'$1'","TotalSamples":48,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_48_'$1'_1_'$4'_HC","LoadName":"MG4R_v2_'$ENVIRON'_SF_48_'$1'","TotalSamples":48,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait

python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_48_'$1'_2_'$4'_HP","LoadName":"MG4R_v2_'$ENVIRON'_SF_48_'$1'","TotalSamples":48,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_48_'$1'_2_'$4'_RS","LoadName":"MG4R_v2_'$ENVIRON'_SF_48_'$1'","TotalSamples":48,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_48_'$1'_2_'$4'_FS","LoadName":"MG4R_v2_'$ENVIRON'_SF_48_'$1'","TotalSamples":48,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_48_'$1'_2_'$4'_RN","LoadName":"MG4R_v2_'$ENVIRON'_SF_48_'$1'","TotalSamples":48,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_48_'$1'_2_'$4'_HC","LoadName":"MG4R_v2_'$ENVIRON'_SF_48_'$1'","TotalSamples":48,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait

python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_48_'$1'_3_'$4'_HP","LoadName":"MG4R_v2_'$ENVIRON'_SF_48_'$1'","TotalSamples":48,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_48_'$1'_3_'$4'_RS","LoadName":"MG4R_v2_'$ENVIRON'_SF_48_'$1'","TotalSamples":48,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_48_'$1'_3_'$4'_FS","LoadName":"MG4R_v2_'$ENVIRON'_SF_48_'$1'","TotalSamples":48,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_48_'$1'_3_'$4'_RN","LoadName":"MG4R_v2_'$ENVIRON'_SF_48_'$1'","TotalSamples":48,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"EnvConfig":"MG4R_v2_'$ENVIRON'.json","RunName":"MG4R_v2_'$ENVIRON'_SFH_48_'$1'_3_'$4'_HC","LoadName":"MG4R_v2_'$ENVIRON'_SF_48_'$1'","TotalSamples":48,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait
