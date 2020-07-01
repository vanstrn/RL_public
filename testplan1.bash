echo $1
echo $2
echo $3
echo $4
echo $5

# python SF_v2_action.py -f CTFP20_SFa.json -c '{"RunName":"CTFP20_256_'$1'"}' -n '{"SFSize":"256"}'
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFP20_FSv2_256_'$1'_'$2'_NOpt16_FS'$4'_E21_L51","LoadName":"CTFP20_256_'$1'","TotalSamples":256,"Selection":"'$3'","NumOptions":16,"NetworkHPs":{"EntropyBeta":0.01,"LR":0.00001,"FS":'$4'}}'
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFP20_FSv2_256_'$1'_'$2'_NOpt32_FS'$4'_E21_L51","LoadName":"CTFP20_256_'$1'","TotalSamples":256,"Selection":"'$3'","NumOptions":32,"NetworkHPs":{"EntropyBeta":0.01,"LR":0.00001,"FS":'$4'}}'
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFP20_FSv2_256_'$1'_'$2'_NOpt40_FS'$4'_E21_L51","LoadName":"CTFP20_256_'$1'","TotalSamples":256,"Selection":"'$3'","NumOptions":40,"NetworkHPs":{"EntropyBeta":0.01,"LR":0.00001,"FS":'$4'}}'

python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFP20_FSv2_256_'$1'_'$2'_NOpt16_FS'$4'_E25_L55","LoadName":"CTFP20_256_'$1'","TotalSamples":256,"Selection":"'$3'","NumOptions":16,"NetworkHPs":{"EntropyBeta":0.05,"LR":0.00005,"FS":'$4'}}'
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFP20_FSv2_256_'$1'_'$2'_NOpt32_FS'$4'_E25_L55","LoadName":"CTFP20_256_'$1'","TotalSamples":256,"Selection":"'$3'","NumOptions":32,"NetworkHPs":{"EntropyBeta":0.05,"LR":0.00005,"FS":'$4'}}'
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFP20_FSv2_256_'$1'_'$2'_NOpt40_FS'$4'_E25_L55","LoadName":"CTFP20_256_'$1'","TotalSamples":256,"Selection":"'$3'","NumOptions":40,"NetworkHPs":{"EntropyBeta":0.05,"LR":0.00005,"FS":'$4'}}'
