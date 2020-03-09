python AsynchExec.py -f MG4RSS_AE.json -c '{"RunName":"MG4RSS_AE_DNN_v3_600_400_512_1f","NetworkConfig":"MG4RSS_AE_DNN_v3","NetworkHPs":{"State LR": 1E-4,"Optimizer":"Adam"}}' -n '{"DefaultParams":{"SFSize":512,"Hidden1":600,"Hidden2":400}}' &
python AsynchExec.py -f MG4RSS_AE.json -c '{"RunName":"MG4RSS_AE_DNN_v3_600_400_256_1f","NetworkConfig":"MG4RSS_AE_DNN_v3","NetworkHPs":{"State LR": 1E-4,"Optimizer":"Adam"}}' -n '{"DefaultParams":{"SFSize":256,"Hidden1":600,"Hidden2":400}}' &
python AsynchExec.py -f MG4RSS_AE_4f.json -c '{"RunName":"MG4RSS_AE_DNN_v3_600_400_512_4f","NetworkConfig":"MG4RSS_AE_DNN_v2","NetworkHPs":{"State LR": 1E-4,"Optimizer":"Adam"}}' -n '{"DefaultParams":{"SFSize":512,"Hidden1":600,"Hidden2":400}}' &
python AsynchExec.py -f MG4RSS_AE_4f.json -c '{"RunName":"MG4RSS_AE_DNN_v3_600_400_256_4f","NetworkConfig":"MG4RSS_AE_DNN_v2","NetworkHPs":{"State LR": 1E-4,"Optimizer":"Adam"}}' -n '{"DefaultParams":{"SFSize":256,"Hidden1":600,"Hidden2":400}}'

python AsynchExec.py -f MG4RP_AE.json -c '{"RunName":"MG4RP_AE_CNN_lit","NetworkConfig":"MG4RP_AE_Lit","NetworkHPs":{"State LR": 1E-4,"Optimizer":"Adam"}}'
