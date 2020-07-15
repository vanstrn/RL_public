echo "Batch Number" $1
if [ -z "$2" ]
  then
    echo "Device argument not supplied defaulting to CPU"
    DEVICE=cpu
  else
    DEVICE=$2
fi
echo "Using device:" $DEVICE

python SF_v2_action.py -f CTFP20_SFa.json -c '{"RunName":"CTF1v0_SFa_256_'$1'"}' -n '{"DefaultParams":{"SFaize":"256"}}' -p /$DEVICE:0

python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTF1v0_SFaH_256_'$1'_1_PPO_HP","LoadName":"CTF1v0_SFa_256_'$1'","TotalSamples":256,"Selection":"Hull_pca","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":256,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTF1v0_SFaH_256_'$1'_1_PPO_RS","LoadName":"CTF1v0_SFa_256_'$1'","TotalSamples":256,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":256,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTF1v0_SFaH_256_'$1'_1_PPO_FS","LoadName":"CTF1v0_SFa_256_'$1'","TotalSamples":256,"Selection":"First","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":256,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTF1v0_SFaH_256_'$1'_1_PPO_RN","LoadName":"CTF1v0_SFa_256_'$1'","TotalSamples":256,"Selection":"Random","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":256,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTF1v0_SFaH_256_'$1'_1_PPO_HC","LoadName":"CTF1v0_SFa_256_'$1'","TotalSamples":256,"Selection":"Hull_cluster","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":256,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait
