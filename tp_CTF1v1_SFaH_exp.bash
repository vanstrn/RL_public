echo "Batch Number" $1
if [ -z "$2" ]
  then
    echo "Device argument not supplied defaulting to CPU"
    DEVICE=cpu
  else
    DEVICE=$2
fi
echo "Using device:" $DEVICE

# python SF_v2_action.py -f CTF20_SFa.json -c '{"RunName":"CTF1v1_SFa_512_'$1'"}' -n '{"DefaultParams":{"SFaize":"512"}}' -p /$DEVICE:0

# python SingleExec_v4.py -f CTF_SFaH.json -c '{"RunName":"CTF1v1_SFaH_512_'$1'_1_PPO_HP","LoadName":"CTF1v1_SFa_512_'$1'","TotalSamples":512,"Selection":"Hull_pca","NumOptions":24,
#     "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f CTF_SFaH.json -c '{"RunName":"CTF1v1_SFaH_512_'$1'_1_PPO_RS_ENT5E6_2","LoadName":"CTF1v1_SFa_512_'$1'","TotalSamples":512,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00001,"LR Entropy":0.000005,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
python SingleExec_v4.py -f CTF_SFaH.json -c '{"RunName":"CTF1v1_SFaH_512_'$1'_1_PPO_RS_ENT1E5_2","LoadName":"CTF1v1_SFa_512_'$1'","TotalSamples":512,"Selection":"Random_sampling","NumOptions":24,
    "NetworkHPs":{"EntropyBeta":0.1,"LR Actor":0.00001,"LR Entropy":0.00001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
# python SingleExec_v4.py -f CTF_SFaH.json -c '{"RunName":"CTF1v1_SFaH_512_'$1'_1_PPO_FS_2","LoadName":"CTF1v1_SFa_512_'$1'","TotalSamples":512,"Selection":"First","NumOptions":24,
#     "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
# python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFaH.json -c '{"RunName":"CTF1v1_SFaH_512_'$1'_1_PPO_RS","LoadName":"CTF1v1_SFa_512_'$1'","TotalSamples":512,"Selection":"Random_sampling","NumOptions":24,
#     "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
# python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFaH.json -c '{"RunName":"CTF1v1_SFaH_512_'$1'_1_PPO_FS","LoadName":"CTF1v1_SFa_512_'$1'","TotalSamples":512,"Selection":"First","NumOptions":24,
#     "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
# python SingleExec_v4.py -f CTF_SFaH.json -c '{"RunName":"CTF1v1_SFaH_512_'$1'_1_PPO_RN","LoadName":"CTF1v1_SFa_512_'$1'","TotalSamples":512,"Selection":"Random","NumOptions":24,
#     "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
# python SingleExec_v4.py -f CTF20_SFaH.json -c '{"RunName":"CTF1v1_SFaH_512_'$1'_1_PPO_HC","LoadName":"CTF1v1_SFa_512_'$1'","TotalSamples":512,"Selection":"Hull_cluster","NumOptions":24,
#     "NetworkHPs":{"EntropyBeta":0.0,"LR Actor":0.00005,"LR Entropy":0.0000001,"BatchSize":512,"MinibatchSize":32,"CriticBeta":0.5,"Gamma":0.98,"lambda":0.98,"Epochs":1}}' -p /$DEVICE:0 &
wait
