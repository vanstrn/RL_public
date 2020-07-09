echo "Batch Number" $1
if [ -z "$2" ]
  then
    echo "Device argument not supplied defaulting to CPU"
    DEVICE=cpu
  else
    DEVICE=$2
fi
echo "Using device:" $DEVICE

python SF_v2.py -f MG4R_v2_SF.json -c '{"RunName":"MG4R_v2_SF_128_'$1'_Hyperparameter"}' -n '{"SFSize":"128"}' -p /$DEVICE:0

#Options Test
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_OPT36_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":36,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_OPT32_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":32,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_OPT28_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":28,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_OPT24_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_OPT20_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":20,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_OPT16_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":16,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_OPT12_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":12,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
wait


#Gamma Experiment
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_Gamma9E1_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_Gamma95E1_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.95,"lambda":0.95,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_Gamma98E1_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_Gamma99E1_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.99,"lambda":0.99,"Epochs":1}}' -p /$DEVICE:0 &
wait


#Entropy Experiment
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_ENTLR1E6_BETA1E1_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_ENTLR1E6_BETA1E2_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.01,"LR Actor":0.00005,"LR Entropy":0.000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_ENTLR1E6_BETA1E3_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.001,"LR Actor":0.00005,"LR Entropy":0.000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_ENTLR1E6_BETA1E4_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.0001,"LR Actor":0.00005,"LR Entropy":0.000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_ENTLR1E6_BETA1E5_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.00001,"LR Actor":0.00005,"LR Entropy":0.000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
wait


python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_ENTLR1E7_BETA1E1_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_ENTLR1E7_BETA1E2_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.01,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_ENTLR1E7_BETA1E3_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.001,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_ENTLR1E7_BETA1E4_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.0001,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_ENTLR1E7_BETA1E5_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.00001,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
wait


# Learning Rate Tests
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_ALR1E4_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.0001,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_ALR5E5_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_ALR1E5_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00001,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_ALR5E6_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.000005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_ALR1E6_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.000001,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
wait


#BatchSize Experiments
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_BS1024_MB32_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_BS1024_MB64_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":64,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_BS512_MB32_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_BS512_MB64_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":512,"MinibatchSize":64,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_BS256_MB32_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":256,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_BS256_MB64_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":256,"MinibatchSize":64,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1}}' -p /$DEVICE:0 &
wait


#Fixed Step Experiments
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_FS1_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1,"FS":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_FS2_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1,"FS":2}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_FS3_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1,"FS":3}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_FS4_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1,"FS":4}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_FS5_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1,"FS":5}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_FS6_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1,"FS":6}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_3_SFH_FS7_'$1'","LoadName":"MG4R_v2_SF_128_'$1'_Hyperparameter","TotalSamples":128,"Selection":"Hull_pca","NumOptions":24,
  "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0,"BatchSize":1024,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.9,"lambda":0.9,"Epochs":1,"FS":7}}' -p /$DEVICE:0 &
wait
