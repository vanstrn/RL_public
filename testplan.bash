echo $1

python SF_v2.py -f CTFP30_SF.json -c '{"RunName":"CTFP30_512_'$1'"}' -n '{"SFSize":"512"}'
python SingleExec_v4.py -f CTFP30_SFH.json -c '{"RunName":"CTFP30_512_'$1'_FS","LoadName":"CTFP30_512_'$1'","TotalSamples":512,"Selection":"First"}'
python SingleExec_v4.py -f CTFP30_SFH.json -c '{"RunName":"CTFP30_512_'$1'_RN","LoadName":"CTFP30_512_'$1'","TotalSamples":512,"Selection":"Random"}'
python SingleExec_v4.py -f CTFP30_SFH.json -c '{"RunName":"CTFP30_512_'$1'_RE","LoadName":"CTFP30_512_'$1'","TotalSamples":512,"Selection":"Random_sampling"}'
python SingleExec_v4.py -f CTFP30_SFH.json -c '{"RunName":"CTFP30_512_'$1'_HP","LoadName":"CTFP30_512_'$1'","TotalSamples":512,"Selection":"Hull_pca"}'
python SingleExec_v4.py -f CTFP30_SFH.json -c '{"RunName":"CTFP30_512_'$1'_HC","LoadName":"CTFP30_512_'$1'","TotalSamples":512,"Selection":"Hull_cluster"}'

python SF_v2.py -f CTFP30_SF.json -c '{"RunName":"CTFP30_384_'$1'"}' -n '{"SFSize":"384"}'
python SingleExec_v4.py -f CTFP30_SFH.json -c '{"RunName":"CTFP30_384_'$1'_FS","LoadName":"CTFP30_384_'$1'","TotalSamples":384,"Selection":"First"}'
python SingleExec_v4.py -f CTFP30_SFH.json -c '{"RunName":"CTFP30_384_'$1'_RN","LoadName":"CTFP30_384_'$1'","TotalSamples":384,"Selection":"Random"}'
python SingleExec_v4.py -f CTFP30_SFH.json -c '{"RunName":"CTFP30_384_'$1'_RE","LoadName":"CTFP30_384_'$1'","TotalSamples":384,"Selection":"Random_sampling"}'
python SingleExec_v4.py -f CTFP30_SFH.json -c '{"RunName":"CTFP30_384_'$1'_HP","LoadName":"CTFP30_384_'$1'","TotalSamples":384,"Selection":"Hull_pca"}'
python SingleExec_v4.py -f CTFP30_SFH.json -c '{"RunName":"CTFP30_384_'$1'_HC","LoadName":"CTFP30_384_'$1'","TotalSamples":384,"Selection":"Hull_cluster"}'

python SF_v2.py -f CTFP30_SF.json -c '{"RunName":"CTFP30_256_'$1'"}' -n '{"SFSize":"256"}'
python SingleExec_v4.py -f CTFP30_SFH.json -c '{"RunName":"CTFP30_256_'$1'_FS","LoadName":"CTFP30_256_'$1'","TotalSamples":256,"Selection":"First"}'
python SingleExec_v4.py -f CTFP30_SFH.json -c '{"RunName":"CTFP30_256_'$1'_RN","LoadName":"CTFP30_256_'$1'","TotalSamples":256,"Selection":"Random"}'
python SingleExec_v4.py -f CTFP30_SFH.json -c '{"RunName":"CTFP30_256_'$1'_RE","LoadName":"CTFP30_256_'$1'","TotalSamples":256,"Selection":"Random_sampling"}'
python SingleExec_v4.py -f CTFP30_SFH.json -c '{"RunName":"CTFP30_256_'$1'_HP","LoadName":"CTFP30_256_'$1'","TotalSamples":256,"Selection":"Hull_pca"}'
python SingleExec_v4.py -f CTFP30_SFH.json -c '{"RunName":"CTFP30_256_'$1'_HC","LoadName":"CTFP30_256_'$1'","TotalSamples":256,"Selection":"Hull_cluster"}'

python SF_v2.py -f CTFP30_SF.json -c '{"RunName":"CTFP30_128_'$1'"}' -n '{"SFSize":"128"}'
python SingleExec_v4.py -f CTFP30_SFH.json -c '{"RunName":"CTFP30_128_'$1'_FS","LoadName":"CTFP30_128_'$1'","TotalSamples":128,"Selection":"First"}'
python SingleExec_v4.py -f CTFP30_SFH.json -c '{"RunName":"CTFP30_128_'$1'_RN","LoadName":"CTFP30_128_'$1'","TotalSamples":128,"Selection":"Random"}'
python SingleExec_v4.py -f CTFP30_SFH.json -c '{"RunName":"CTFP30_128_'$1'_RE","LoadName":"CTFP30_128_'$1'","TotalSamples":128,"Selection":"Random_sampling"}'
python SingleExec_v4.py -f CTFP30_SFH.json -c '{"RunName":"CTFP30_128_'$1'_HP","LoadName":"CTFP30_128_'$1'","TotalSamples":128,"Selection":"Hull_pca"}'
python SingleExec_v4.py -f CTFP30_SFH.json -c '{"RunName":"CTFP30_128_'$1'_HC","LoadName":"CTFP30_128_'$1'","TotalSamples":128,"Selection":"Hull_cluster"}'


python SF_v2_action.py -f CTFP30_SFa.json -c '{"RunName":"CTFPa30_512_'$1'"}' -n '{"SFSize":"512"}'
python SingleExec_v4.py -f CTFP30_SFaH.json -c '{"RunName":"CTFPa30_512_'$1'_FS","LoadName":"CTFPa30_512_'$1'","TotalSamples":512,"Selection":"First"}'
python SingleExec_v4.py -f CTFP30_SFaH.json -c '{"RunName":"CTFPa30_512_'$1'_RN","LoadName":"CTFPa30_512_'$1'","TotalSamples":512,"Selection":"Random"}'
python SingleExec_v4.py -f CTFP30_SFaH.json -c '{"RunName":"CTFPa30_512_'$1'_RE","LoadName":"CTFPa30_512_'$1'","TotalSamples":512,"Selection":"Random_sampling"}'
python SingleExec_v4.py -f CTFP30_SFaH.json -c '{"RunName":"CTFPa30_512_'$1'_HP","LoadName":"CTFPa30_512_'$1'","TotalSamples":512,"Selection":"Hull_pca"}'
python SingleExec_v4.py -f CTFP30_SFaH.json -c '{"RunName":"CTFPa30_512_'$1'_HC","LoadName":"CTFPa30_512_'$1'","TotalSamples":512,"Selection":"Hull_cluster"}'

python SF_v2_action.py -f CTFP30_SFa.json -c '{"RunName":"CTFPa30_384_'$1'"}' -n '{"SFSize":"384"}'
python SingleExec_v4.py -f CTFP30_SFaH.json -c '{"RunName":"CTFPa30_384_'$1'_FS","LoadName":"CTFPa30_384_'$1'","TotalSamples":384,"Selection":"First"}'
python SingleExec_v4.py -f CTFP30_SFaH.json -c '{"RunName":"CTFPa30_384_'$1'_RN","LoadName":"CTFPa30_384_'$1'","TotalSamples":384,"Selection":"Random"}'
python SingleExec_v4.py -f CTFP30_SFaH.json -c '{"RunName":"CTFPa30_384_'$1'_RE","LoadName":"CTFPa30_384_'$1'","TotalSamples":384,"Selection":"Random_sampling"}'
python SingleExec_v4.py -f CTFP30_SFaH.json -c '{"RunName":"CTFPa30_384_'$1'_HP","LoadName":"CTFPa30_384_'$1'","TotalSamples":384,"Selection":"Hull_pca"}'
python SingleExec_v4.py -f CTFP30_SFaH.json -c '{"RunName":"CTFPa30_384_'$1'_HC","LoadName":"CTFPa30_384_'$1'","TotalSamples":384,"Selection":"Hull_cluster"}'

python SF_v2_action.py -f CTFP30_SFa.json -c '{"RunName":"CTFPa30_256_'$1'"}' -n '{"SFSize":"256"}'
python SingleExec_v4.py -f CTFP30_SFaH.json -c '{"RunName":"CTFPa30_256_'$1'_FS","LoadName":"CTFPa30_256_'$1'","TotalSamples":256,"Selection":"First"}'
python SingleExec_v4.py -f CTFP30_SFaH.json -c '{"RunName":"CTFPa30_256_'$1'_RN","LoadName":"CTFPa30_256_'$1'","TotalSamples":256,"Selection":"Random"}'
python SingleExec_v4.py -f CTFP30_SFaH.json -c '{"RunName":"CTFPa30_256_'$1'_RE","LoadName":"CTFPa30_256_'$1'","TotalSamples":256,"Selection":"Random_sampling"}'
python SingleExec_v4.py -f CTFP30_SFaH.json -c '{"RunName":"CTFPa30_256_'$1'_HP","LoadName":"CTFPa30_256_'$1'","TotalSamples":256,"Selection":"Hull_pca"}'
python SingleExec_v4.py -f CTFP30_SFaH.json -c '{"RunName":"CTFPa30_256_'$1'_HC","LoadName":"CTFPa30_256_'$1'","TotalSamples":256,"Selection":"Hull_cluster"}'

python SF_v2_action.py -f CTFP30_SFa.json -c '{"RunName":"CTFPa30_128_'$1'"}' -n '{"SFSize":"128"}'
python SingleExec_v4.py -f CTFP30_SFaH.json -c '{"RunName":"CTFPa30_128_'$1'_FS","LoadName":"CTFPa30_128_'$1'","TotalSamples":128,"Selection":"First"}'
python SingleExec_v4.py -f CTFP30_SFaH.json -c '{"RunName":"CTFPa30_128_'$1'_RN","LoadName":"CTFPa30_128_'$1'","TotalSamples":128,"Selection":"Random"}'
python SingleExec_v4.py -f CTFP30_SFaH.json -c '{"RunName":"CTFPa30_128_'$1'_RE","LoadName":"CTFPa30_128_'$1'","TotalSamples":128,"Selection":"Random_sampling"}'
python SingleExec_v4.py -f CTFP30_SFaH.json -c '{"RunName":"CTFPa30_128_'$1'_HP","LoadName":"CTFPa30_128_'$1'","TotalSamples":128,"Selection":"Hull_pca"}'
python SingleExec_v4.py -f CTFP30_SFaH.json -c '{"RunName":"CTFPa30_128_'$1'_HC","LoadName":"CTFPa30_128_'$1'","TotalSamples":128,"Selection":"Hull_cluster"}'


python SF_v2.py -f CTFP20_SF.json -c '{"RunName":"CTFP20_256_'$1'"}' -n '{"SFSize":"256"}'
python SingleExec_v4.py -f CTFP20_SFH.json -c '{"RunName":"CTFP20_256_'$1'_FS","LoadName":"CTFP20_256_'$1'","TotalSamples":256,"Selection":"First"}'
python SingleExec_v4.py -f CTFP20_SFH.json -c '{"RunName":"CTFP20_256_'$1'_RN","LoadName":"CTFP20_256_'$1'","TotalSamples":256,"Selection":"Random"}'
python SingleExec_v4.py -f CTFP20_SFH.json -c '{"RunName":"CTFP20_256_'$1'_RE","LoadName":"CTFP20_256_'$1'","TotalSamples":256,"Selection":"Random_sampling"}'
python SingleExec_v4.py -f CTFP20_SFH.json -c '{"RunName":"CTFP20_256_'$1'_HP","LoadName":"CTFP20_256_'$1'","TotalSamples":256,"Selection":"Hull_pca"}'
python SingleExec_v4.py -f CTFP20_SFH.json -c '{"RunName":"CTFP20_256_'$1'_HC","LoadName":"CTFP20_256_'$1'","TotalSamples":256,"Selection":"Hull_cluster"}'

python SF_v2.py -f CTFP20_SF.json -c '{"RunName":"CTFP20_192_'$1'"}' -n '{"SFSize":"192"}'
python SingleExec_v4.py -f CTFP20_SFH.json -c '{"RunName":"CTFP20_192_'$1'_FS","LoadName":"CTFP20_192_'$1'","TotalSamples":192,"Selection":"First"}'
python SingleExec_v4.py -f CTFP20_SFH.json -c '{"RunName":"CTFP20_192_'$1'_RN","LoadName":"CTFP20_192_'$1'","TotalSamples":192,"Selection":"Random"}'
python SingleExec_v4.py -f CTFP20_SFH.json -c '{"RunName":"CTFP20_192_'$1'_RE","LoadName":"CTFP20_192_'$1'","TotalSamples":192,"Selection":"Random_sampling"}'
python SingleExec_v4.py -f CTFP20_SFH.json -c '{"RunName":"CTFP20_192_'$1'_HP","LoadName":"CTFP20_192_'$1'","TotalSamples":192,"Selection":"Hull_pca"}'
python SingleExec_v4.py -f CTFP20_SFH.json -c '{"RunName":"CTFP20_192_'$1'_HC","LoadName":"CTFP20_192_'$1'","TotalSamples":192,"Selection":"Hull_cluster"}'

python SF_v2.py -f CTFP20_SF.json -c '{"RunName":"CTFP20_128_'$1'"}' -n '{"SFSize":"128"}'
python SingleExec_v4.py -f CTFP20_SFH.json -c '{"RunName":"CTFP20_128_'$1'_FS","LoadName":"CTFP20_128_'$1'","TotalSamples":128,"Selection":"First"}'
python SingleExec_v4.py -f CTFP20_SFH.json -c '{"RunName":"CTFP20_128_'$1'_RN","LoadName":"CTFP20_128_'$1'","TotalSamples":128,"Selection":"Random"}'
python SingleExec_v4.py -f CTFP20_SFH.json -c '{"RunName":"CTFP20_128_'$1'_RE","LoadName":"CTFP20_128_'$1'","TotalSamples":128,"Selection":"Random_sampling"}'
python SingleExec_v4.py -f CTFP20_SFH.json -c '{"RunName":"CTFP20_128_'$1'_HP","LoadName":"CTFP20_128_'$1'","TotalSamples":128,"Selection":"Hull_pca"}'
python SingleExec_v4.py -f CTFP20_SFH.json -c '{"RunName":"CTFP20_128_'$1'_HC","LoadName":"CTFP20_128_'$1'","TotalSamples":128,"Selection":"Hull_cluster"}'

python SF_v2.py -f CTFP20_SF.json -c '{"RunName":"CTFP20_96_'$1'"}' -n '{"SFSize":"96"}'
python SingleExec_v4.py -f CTFP20_SFH.json -c '{"RunName":"CTFP20_96_'$1'_FS","LoadName":"CTFP20_96_'$1'","TotalSamples":96,"Selection":"First"}'
python SingleExec_v4.py -f CTFP20_SFH.json -c '{"RunName":"CTFP20_96_'$1'_RN","LoadName":"CTFP20_96_'$1'","TotalSamples":96,"Selection":"Random"}'
python SingleExec_v4.py -f CTFP20_SFH.json -c '{"RunName":"CTFP20_96_'$1'_RE","LoadName":"CTFP20_96_'$1'","TotalSamples":96,"Selection":"Random_sampling"}'
python SingleExec_v4.py -f CTFP20_SFH.json -c '{"RunName":"CTFP20_96_'$1'_HP","LoadName":"CTFP20_96_'$1'","TotalSamples":96,"Selection":"Hull_pca"}'
python SingleExec_v4.py -f CTFP20_SFH.json -c '{"RunName":"CTFP20_96_'$1'_HC","LoadName":"CTFP20_96_'$1'","TotalSamples":96,"Selection":"Hull_cluster"}'


python SF_v2.py -f CTFP20_SFa.json -c '{"RunName":"CTFPa20_256_'$1'"}' -n '{"SFSize":"256"}'
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFPa20_256_'$1'_FS","LoadName":"CTFP20_256_'$1'","TotalSamples":256,"Selection":"First"}'
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFPa20_256_'$1'_RN","LoadName":"CTFP20_256_'$1'","TotalSamples":256,"Selection":"Random"}'
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFPa20_256_'$1'_RE","LoadName":"CTFP20_256_'$1'","TotalSamples":256,"Selection":"Random_sampling"}'
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFPa20_256_'$1'_HP","LoadName":"CTFP20_256_'$1'","TotalSamples":256,"Selection":"Hull_pca"}'
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFPa20_256_'$1'_HC","LoadName":"CTFP20_256_'$1'","TotalSamples":256,"Selection":"Hull_cluster"}'

python SF_v2.py -f CTFP20_SFa.json -c '{"RunName":"CTFPa20_192_'$1'"}' -n '{"SFSize":"192"}'
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFPa20_192_'$1'_FS","LoadName":"CTFP20_192_'$1'","TotalSamples":192,"Selection":"First"}'
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFPa20_192_'$1'_RN","LoadName":"CTFP20_192_'$1'","TotalSamples":192,"Selection":"Random"}'
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFPa20_192_'$1'_RE","LoadName":"CTFP20_192_'$1'","TotalSamples":192,"Selection":"Random_sampling"}'
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFPa20_192_'$1'_HP","LoadName":"CTFP20_192_'$1'","TotalSamples":192,"Selection":"Hull_pca"}'
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFPa20_192_'$1'_HC","LoadName":"CTFP20_192_'$1'","TotalSamples":192,"Selection":"Hull_cluster"}'

python SF_v2.py -f CTFP20_SFa.json -c '{"RunName":"CTFPa20_128_'$1'"}' -n '{"SFSize":"128"}'
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFPa20_128_'$1'_FS","LoadName":"CTFP20_128_'$1'","TotalSamples":128,"Selection":"First"}'
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFPa20_128_'$1'_RN","LoadName":"CTFP20_128_'$1'","TotalSamples":128,"Selection":"Random"}'
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFPa20_128_'$1'_RE","LoadName":"CTFP20_128_'$1'","TotalSamples":128,"Selection":"Random_sampling"}'
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFPa20_128_'$1'_HP","LoadName":"CTFP20_128_'$1'","TotalSamples":128,"Selection":"Hull_pca"}'
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFPa20_128_'$1'_HC","LoadName":"CTFP20_128_'$1'","TotalSamples":128,"Selection":"Hull_cluster"}'

python SF_v2.py -f CTFP20_SFa.json -c '{"RunName":"CTFPa20_96_'$1'"}' -n '{"SFSize":"96"}'
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFPa20_96_'$1'_FS","LoadName":"CTFP20_96_'$1'","TotalSamples":96,"Selection":"First"}'
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFPa20_96_'$1'_RN","LoadName":"CTFP20_96_'$1'","TotalSamples":96,"Selection":"Random"}'
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFPa20_96_'$1'_RE","LoadName":"CTFP20_96_'$1'","TotalSamples":96,"Selection":"Random_sampling"}'
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFPa20_96_'$1'_HP","LoadName":"CTFP20_96_'$1'","TotalSamples":96,"Selection":"Hull_pca"}'
python SingleExec_v4.py -f CTFP20_SFaH.json -c '{"RunName":"CTFPa20_96_'$1'_HC","LoadName":"CTFP20_96_'$1'","TotalSamples":96,"Selection":"Hull_cluster"}'
