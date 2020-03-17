
python AE_Analysis.py -f MG4R_AE_Analysis.json -c '{"RunName":"MG4R_AE_DNN_v3_Adam_4_600_400_512","NetworkConfig":"MG4R_AE_DNN_v3","NetworkHPs":{"State LR": 1E-4,"Optimizer":"Adam"}}' -n '{"DefaultParams":{"SFSize":512,"Hidden1":600,"Hidden2":400}}' &
python AE_Analysis.py -f MG4R_AE_Analysis.json -c '{"RunName":"MG4R_AE_DNN_v3_Adam_4_600_400_1024","NetworkConfig":"MG4R_AE_DNN_v3","NetworkHPs":{"State LR": 1E-4,"Optimizer":"Adam"}}' -n '{"DefaultParams":{"SFSize":1024,"Hidden1":600,"Hidden2":400}}' &
python AE_Analysis.py -f MG4R_AE_Analysis.json -c '{"RunName":"MG4R_AE_DNN_v3_Adam_5_600_400_512","NetworkConfig":"MG4R_AE_DNN_v3","NetworkHPs":{"State LR": 1E-5,"Optimizer":"Adam"}}' -n '{"DefaultParams":{"SFSize":512,"Hidden1":600,"Hidden2":400}}' &
python AE_Analysis.py -f MG4R_AE_Analysis.json -c '{"RunName":"MG4R_AE_DNN_v3_Adam_5_600_400_1024","NetworkConfig":"MG4R_AE_DNN_v3","NetworkHPs":{"State LR": 1E-5,"Optimizer":"Adam"}}' -n '{"DefaultParams":{"SFSize":1024,"Hidden1":600,"Hidden2":400}}'