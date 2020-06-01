Trial=($1)
#SH Training:
python SF_CTF_v2.py -f CTF_SF_v2_1v1.json -c '{"RunName":"CTF_SF_'$1'"}'

# #SF learning
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_SF_'$1'","RunName":"CTF_1v1_HLP_'$1'_1","Selection":"Hull_pca"}'
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_SF_'$1'","RunName":"CTF_1v1_HCL_'$1'_1","Selection":"Hull_cluster"}'
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_SF_'$1'","RunName":"CTF_1v1_FST_'$1'_1","Selection":"First"}'
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_SF_'$1'","RunName":"CTF_1v1_RND_'$1'_1","Selection":"Random"}'
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_SF_'$1'","RunName":"CTF_1v1_HLT_'$1'_1","Selection":"Hull_tsne"}'
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_SF_'$1'","RunName":"CTF_1v1_HLP_'$1'_2","Selection":"Hull_pca"}'
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_SF_'$1'","RunName":"CTF_1v1_HCL_'$1'_2","Selection":"Hull_cluster"}'
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_SF_'$1'","RunName":"CTF_1v1_FST_'$1'_2","Selection":"First"}'
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_SF_'$1'","RunName":"CTF_1v1_RND_'$1'_2","Selection":"Random"}'
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_SF_'$1'","RunName":"CTF_1v1_HLT_'$1'_2","Selection":"Hull_tsne"}'
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_SF_'$1'","RunName":"CTF_1v1_HLP_'$1'_3","Selection":"Hull_pca"}'
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_SF_'$1'","RunName":"CTF_1v1_HCL_'$1'_3","Selection":"Hull_cluster"}'
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_SF_'$1'","RunName":"CTF_1v1_FST_'$1'_3","Selection":"First"}'
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_SF_'$1'","RunName":"CTF_1v1_RND_'$1'_3","Selection":"Random"}'
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_SF_'$1'","RunName":"CTF_1v1_HLT_'$1'_3","Selection":"Hull_tsne"}'


#Baseline learning
python SingleExec_v2.py -f CTF_PPO.json -c '{"RunName":"CTF_1v1_Baseline_'$1'_1"}'
python SingleExec_v2.py -f CTF_PPO.json -c '{"RunName":"CTF_1v1_Baseline_'$1'_2"}'
python SingleExec_v2.py -f CTF_PPO.json -c '{"RunName":"CTF_1v1_Baseline_'$1'_3"}'
python SingleExec_v2.py -f CTF_PPO.json -c '{"RunName":"CTF_1v1_Faster_Baseline_'$1'_1"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_faster.ini"}}'
python SingleExec_v2.py -f CTF_PPO.json -c '{"RunName":"CTF_1v1_Slower_Baseline_'$1'_1"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_slower.ini"}}'
python SingleExec_v2.py -f CTF_PPO.json -c '{"RunName":"CTF_1v1_Faster_Baseline_'$1'_2"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_faster.ini"}}'
python SingleExec_v2.py -f CTF_PPO.json -c '{"RunName":"CTF_1v1_Slower_Baseline_'$1'_2"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_slower.ini"}}'
python SingleExec_v2.py -f CTF_PPO.json -c '{"RunName":"CTF_1v1_Faster_Baseline_'$1'_3"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_faster.ini"}}'
python SingleExec_v2.py -f CTF_PPO.json -c '{"RunName":"CTF_1v1_Slower_Baseline_'$1'_3"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_slower.ini"}}'

# #Adaptation Baselines
python SingleExec_v2.py -f CTF_PPO.json -c '{"LoadName":"CTF_1v1_Baseline_'$1'_1","RunName":"CTF_1v1_Faster_Adaptation_'$1'_1"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_faster.ini"}}'
python SingleExec_v2.py -f CTF_PPO.json -c '{"LoadName":"CTF_1v1_Baseline_'$1'_1","RunName":"CTF_1v1_Slower_Adaptation_'$1'_1"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_slower.ini"}}'
python SingleExec_v2.py -f CTF_PPO.json -c '{"LoadName":"CTF_1v1_Baseline_'$1'_2","RunName":"CTF_1v1_Faster_Adaptation_'$1'_2"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_faster.ini"}}'
python SingleExec_v2.py -f CTF_PPO.json -c '{"LoadName":"CTF_1v1_Baseline_'$1'_2","RunName":"CTF_1v1_Faster_Adaptation_'$1'_2"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_faster.ini"}}'
python SingleExec_v2.py -f CTF_PPO.json -c '{"LoadName":"CTF_1v1_Baseline_'$1'_3","RunName":"CTF_1v1_Slower_Adaptation_'$1'_3"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_slower.ini"}}'
python SingleExec_v2.py -f CTF_PPO.json -c '{"LoadName":"CTF_1v1_Baseline_'$1'_3","RunName":"CTF_1v1_Slower_Adaptation_'$1'_3"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_slower.ini"}}'

# #SF Adapation
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_HLP_'$1'","RunName":"CTF_1v1_HLP_Faster_'$1'_1","Selection":"Hull_pca"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_faster.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_HCL_'$1'","RunName":"CTF_1v1_HCL_Faster_'$1'_1","Selection":"Hull_cluster"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_faster.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_FST_'$1'","RunName":"CTF_1v1_FST_Faster_'$1'_1","Selection":"First"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_faster.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_RND_'$1'","RunName":"CTF_1v1_RND_Faster_'$1'_1","Selection":"Random"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_faster.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_HLT_'$1'","RunName":"CTF_1v1_HLT_Faster_'$1'_1","Selection":"Hull_tsne"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_faster.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_HLP_'$1'","RunName":"CTF_1v1_HLP_Faster_'$1'_2","Selection":"Hull_pca"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_faster.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_HCL_'$1'","RunName":"CTF_1v1_HCL_Faster_'$1'_2","Selection":"Hull_cluster"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_faster.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_FST_'$1'","RunName":"CTF_1v1_FST_Faster_'$1'_2","Selection":"First"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_faster.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_RND_'$1'","RunName":"CTF_1v1_RND_Faster_'$1'_2","Selection":"Random"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_faster.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_HLT_'$1'","RunName":"CTF_1v1_HLT_Faster_'$1'_2","Selection":"Hull_tsne"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_faster.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_HLP_'$1'","RunName":"CTF_1v1_HLP_Faster_'$1'_3","Selection":"Hull_pca"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_faster.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_HCL_'$1'","RunName":"CTF_1v1_HCL_Faster_'$1'_3","Selection":"Hull_cluster"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_faster.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_FST_'$1'","RunName":"CTF_1v1_FST_Faster_'$1'_3","Selection":"First"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_faster.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_RND_'$1'","RunName":"CTF_1v1_RND_Faster_'$1'_3","Selection":"Random"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_faster.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_HLT_'$1'","RunName":"CTF_1v1_HLT_Faster_'$1'_3","Selection":"Hull_tsne"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_faster.ini"}}' -l

python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_HLP_'$1'","RunName":"CTF_1v1_HLP_Slower_'$1'_1","Selection":"Hull_pca"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_slower.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_HCL_'$1'","RunName":"CTF_1v1_HCL_Slower_'$1'_1","Selection":"Hull_cluster"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_slower.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_FST_'$1'","RunName":"CTF_1v1_FST_Slower_'$1'_1","Selection":"First"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_slower.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_RND_'$1'","RunName":"CTF_1v1_RND_Slower_'$1'_1","Selection":"Random"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_slower.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_HLT_'$1'","RunName":"CTF_1v1_HLT_Slower_'$1'_1","Selection":"Hull_tsne"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_slower.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_HLP_'$1'","RunName":"CTF_1v1_HLP_Slower_'$1'_2","Selection":"Hull_pca"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_slower.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_HCL_'$1'","RunName":"CTF_1v1_HCL_Slower_'$1'_2","Selection":"Hull_cluster"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_slower.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_FST_'$1'","RunName":"CTF_1v1_FST_Slower_'$1'_2","Selection":"First"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_slower.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_RND_'$1'","RunName":"CTF_1v1_RND_Slower_'$1'_2","Selection":"Random"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_slower.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_HLT_'$1'","RunName":"CTF_1v1_HLT_Slower_'$1'_2","Selection":"Hull_tsne"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_slower.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_HLP_'$1'","RunName":"CTF_1v1_HLP_Slower_'$1'_3","Selection":"Hull_pca"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_slower.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_HCL_'$1'","RunName":"CTF_1v1_HCL_Slower_'$1'_3","Selection":"Hull_cluster"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_slower.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_FST_'$1'","RunName":"CTF_1v1_FST_Slower_'$1'_3","Selection":"First"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_slower.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_RND_'$1'","RunName":"CTF_1v1_RND_Slower_'$1'_3","Selection":"Random"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_slower.ini"}}' -l
python SF_Hierarchy_v3_CTF_1v1_v2.py -f CTF_SFH.json -c '{"LoadName":"CTF_1v1_HLT_'$1'","RunName":"CTF_1v1_HLT_Slower_'$1'_3","Selection":"Hull_tsne"}' -e '{"EnvParams":{ "map_size":20,"config_path":"environments/1v1_slower.ini"}}' -l
