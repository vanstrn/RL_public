echo $1
echo $2

python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_SF_128_'$1'_FS_'$2'","LoadName":"MG4R_19_SF_128_'$1'","TotalSamples":128,"Selection":"First"}'
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_SF_128_'$1'_RN_'$2'","LoadName":"MG4R_19_SF_128_'$1'","TotalSamples":128,"Selection":"Random"}'
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_SF_128_'$1'_RE_'$2'","LoadName":"MG4R_19_SF_128_'$1'","TotalSamples":128,"Selection":"Random_sampling"}'
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_SF_128_'$1'_HP_'$2'","LoadName":"MG4R_19_SF_128_'$1'","TotalSamples":128,"Selection":"Hull_pca"}'

python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_SF_96_'$1'_FS_'$2'","LoadName":"MG4R_19_SF_96_'$1'","TotalSamples":96,"Selection":"First"}'
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_SF_96_'$1'_RN_'$2'","LoadName":"MG4R_19_SF_96_'$1'","TotalSamples":96,"Selection":"Random"}'
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_SF_96_'$1'_RE_'$2'","LoadName":"MG4R_19_SF_96_'$1'","TotalSamples":96,"Selection":"Random_sampling"}'
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_SF_96_'$1'_HP_'$2'","LoadName":"MG4R_19_SF_96_'$1'","TotalSamples":96,"Selection":"Hull_pca"}'

python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_SF_80_'$1'_FS_'$2'","LoadName":"MG4R_19_SF_80_'$1'","TotalSamples":80,"Selection":"First"}'
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_SF_80_'$1'_RN_'$2'","LoadName":"MG4R_19_SF_80_'$1'","TotalSamples":80,"Selection":"Random"}'
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_SF_80_'$1'_RE_'$2'","LoadName":"MG4R_19_SF_80_'$1'","TotalSamples":80,"Selection":"Random_sampling"}'
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_SF_80_'$1'_HP_'$2'","LoadName":"MG4R_19_SF_80_'$1'","TotalSamples":80,"Selection":"Hull_pca"}'

python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_SF_64_'$1'_FS_'$2'","LoadName":"MG4R_19_SF_64_'$1'","TotalSamples":64,"Selection":"First"}'
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_SF_64_'$1'_RN_'$2'","LoadName":"MG4R_19_SF_64_'$1'","TotalSamples":64,"Selection":"Random"}'
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_SF_64_'$1'_RE_'$2'","LoadName":"MG4R_19_SF_64_'$1'","TotalSamples":64,"Selection":"Random_sampling"}'
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_SF_64_'$1'_HP_'$2'","LoadName":"MG4R_19_SF_64_'$1'","TotalSamples":64,"Selection":"Hull_pca"}'

python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_SF_48_'$1'_FS_'$2'","LoadName":"MG4R_19_SF_48_'$1'","TotalSamples":48,"Selection":"First"}'
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_SF_48_'$1'_RN_'$2'","LoadName":"MG4R_19_SF_48_'$1'","TotalSamples":48,"Selection":"Random"}'
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_SF_48_'$1'_RE_'$2'","LoadName":"MG4R_19_SF_48_'$1'","TotalSamples":48,"Selection":"Random_sampling"}'
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_SF_48_'$1'_HP_'$2'","LoadName":"MG4R_19_SF_48_'$1'","TotalSamples":48,"Selection":"Hull_pca"}'

python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_SF_32_'$1'_FS_'$2'","LoadName":"MG4R_19_SF_32_'$1'","TotalSamples":32,"Selection":"First"}'
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_SF_32_'$1'_RN_'$2'","LoadName":"MG4R_19_SF_32_'$1'","TotalSamples":32,"Selection":"Random"}'
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_SF_32_'$1'_RE_'$2'","LoadName":"MG4R_19_SF_32_'$1'","TotalSamples":32,"Selection":"Random_sampling"}'
python SingleExec_v4.py -f MG4R_v2_SFH.json -c '{"RunName":"MG4R_v2_SF_32_'$1'_HP_'$2'","LoadName":"MG4R_19_SF_32_'$1'","TotalSamples":32,"Selection":"Hull_pca"}'


# python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFa_128_'$1'_FS_'$2'","LoadName":"MG4R_v2_SFa_128_'$1'","TotalSamples":128,"Selection":"First"}'
# python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFa_128_'$1'_RN_'$2'","LoadName":"MG4R_v2_SFa_128_'$1'","TotalSamples":128,"Selection":"Random"}'
# python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFa_128_'$1'_RE_'$2'","LoadName":"MG4R_v2_SFa_128_'$1'","TotalSamples":128,"Selection":"Random_sampling"}'
# python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFa_128_'$1'_HP_'$2'","LoadName":"MG4R_v2_SFa_128_'$1'","TotalSamples":128,"Selection":"Hull_pca"}'
#
# python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFa_96_'$1'_FS_'$2'","LoadName":"MG4R_v2_SFa_96_'$1'","TotalSamples":96,"Selection":"First"}'
# python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFa_96_'$1'_RN_'$2'","LoadName":"MG4R_v2_SFa_96_'$1'","TotalSamples":96,"Selection":"Random"}'
# python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFa_96_'$1'_RE_'$2'","LoadName":"MG4R_v2_SFa_96_'$1'","TotalSamples":96,"Selection":"Random_sampling"}'
# python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFa_96_'$1'_HP_'$2'","LoadName":"MG4R_v2_SFa_96_'$1'","TotalSamples":96,"Selection":"Hull_pca"}'
#
# python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFa_80_'$1'_FS_'$2'","LoadName":"MG4R_v2_SFa_80_'$1'","TotalSamples":80,"Selection":"First"}'
# python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFa_80_'$1'_RN_'$2'","LoadName":"MG4R_v2_SFa_80_'$1'","TotalSamples":80,"Selection":"Random"}'
# python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFa_80_'$1'_RE_'$2'","LoadName":"MG4R_v2_SFa_80_'$1'","TotalSamples":80,"Selection":"Random_sampling"}'
# python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFa_80_'$1'_HP_'$2'","LoadName":"MG4R_v2_SFa_80_'$1'","TotalSamples":80,"Selection":"Hull_pca"}'
#
# python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFa_64_'$1'_FS_'$2'","LoadName":"MG4R_v2_SFa_64_'$1'","TotalSamples":64,"Selection":"First"}'
# python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFa_64_'$1'_RN_'$2'","LoadName":"MG4R_v2_SFa_64_'$1'","TotalSamples":64,"Selection":"Random"}'
# python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFa_64_'$1'_RE_'$2'","LoadName":"MG4R_v2_SFa_64_'$1'","TotalSamples":64,"Selection":"Random_sampling"}'
# python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFa_64_'$1'_HP_'$2'","LoadName":"MG4R_v2_SFa_64_'$1'","TotalSamples":64,"Selection":"Hull_pca"}'
#
# python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFa_48_'$1'_FS_'$2'","LoadName":"MG4R_v2_SFa_48_'$1'","TotalSamples":48,"Selection":"First"}'
# python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFa_48_'$1'_RN_'$2'","LoadName":"MG4R_v2_SFa_48_'$1'","TotalSamples":48,"Selection":"Random"}'
# python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFa_48_'$1'_RE_'$2'","LoadName":"MG4R_v2_SFa_48_'$1'","TotalSamples":48,"Selection":"Random_sampling"}'
# python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFa_48_'$1'_HP_'$2'","LoadName":"MG4R_v2_SFa_48_'$1'","TotalSamples":48,"Selection":"Hull_pca"}'
#
# python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFa_32_'$1'_FS_'$2'","LoadName":"MG4R_v2_SFa_32_'$1'","TotalSamples":32,"Selection":"First"}'
# python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFa_32_'$1'_RN_'$2'","LoadName":"MG4R_v2_SFa_32_'$1'","TotalSamples":32,"Selection":"Random"}'
# python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFa_32_'$1'_RE_'$2'","LoadName":"MG4R_v2_SFa_32_'$1'","TotalSamples":32,"Selection":"Random_sampling"}'
# python SingleExec_v4.py -f MG4R_v2_SFaH.json -c '{"RunName":"MG4R_v2_SFa_32_'$1'_HP_'$2'","LoadName":"MG4R_v2_SFa_32_'$1'","TotalSamples":32,"Selection":"Hull_pca"}'
