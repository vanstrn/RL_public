echo $1

# python SF_v2.py -f MG4R_19_SF.json -c '{"RunName":"MG4R_19_SF_128_'$1'"}' -n '{"SFSize":"128"}'
# python SF_v2.py -f MG4R_19_SF.json -c '{"RunName":"MG4R_19_SF_96_'$1'"}' -n '{"SFSize":"96"}'
# python SF_v2.py -f MG4R_19_SF.json -c '{"RunName":"MG4R_19_SF_80_'$1'"}' -n '{"SFSize":"80"}'
# python SF_v2.py -f MG4R_19_SF.json -c '{"RunName":"MG4R_19_SF_64_'$1'"}' -n '{"SFSize":"64"}'
# python SF_v2.py -f MG4R_19_SF.json -c '{"RunName":"MG4R_19_SF_48_'$1'"}' -n '{"SFSize":"48"}'
# python SF_v2.py -f MG4R_19_SF.json -c '{"RunName":"MG4R_19_SF_32_'$1'"}' -n '{"SFSize":"32"}'

python SF_v2_action.py -f MG4R_19_SFa.json -c '{"RunName":"MG4R_SFa_19_128_'$1'"}' -n '{"SFSize":"128"}'
python SF_v2_action.py -f MG4R_19_SFa.json -c '{"RunName":"MG4R_SFa_19_96_'$1'"}' -n '{"SFSize":"96"}'
python SF_v2_action.py -f MG4R_19_SFa.json -c '{"RunName":"MG4R_SFa_19_80_'$1'"}' -n '{"SFSize":"80"}'
python SF_v2_action.py -f MG4R_19_SFa.json -c '{"RunName":"MG4R_SFa_19_64_'$1'"}' -n '{"SFSize":"64"}'
python SF_v2_action.py -f MG4R_19_SFa.json -c '{"RunName":"MG4R_SFa_19_48_'$1'"}' -n '{"SFSize":"48"}'
python SF_v2_action.py -f MG4R_19_SFa.json -c '{"RunName":"MG4R_SFa_19_32_'$1'"}' -n '{"SFSize":"32"}'
