#!/bin/bash
# Summary of a real-time run: ./summarize.sh RUN_FOLDER
D=$1
n=$(grep -a -c "FrbServer: beamset" $D/server.log)
echo "chunks processed: $n (~$(echo "$n*2.048/60" | bc -l | cut -c1-4) min of data)"
grep -a "FrbServer: beamset" $D/server.log | grep -a "beams=" | awk -F'beams=' '{split($2,a,"/"); print a[1], a[2]}' | sort -n | awk '{v[NR]=$1; b=$2} END{printf "equivalent beams: min %s  median %s  max %s  (real beams %s) -> %.2fx real time\n", v[1], v[int(NR/2)], v[NR], b, v[int(NR/2)]/b}'
grep -a "Gbps=" $D/server.log | awk -F'Gbps=' '{split($2,a,","); print a[1]}' | sort -n | awk '{v[NR]=$1} END{print "input Gbps: median", v[int(NR/2)]}'
awk -F, 'NR>1{u+=$2; n++; m=($3+0>m)?$3+0:m; p+=$4} END{printf "GPU: mean use %.0f %%, max VRAM %d MiB, mean power %.0f W\n", u/n, m, p/n}' $D/gpu.csv
awk 'NR>2 && $4 ~ /^[0-9]+$/ {us+=$13; sy+=$14; id+=$15; n++; f=$4/1048576; if(min==""||f<min)min=f} END{printf "CPU: us %.0f %%, sy %.0f %%, idle %.0f %%; minimum free RAM %.1f GB\n", us/n, sy/n, id/n, min}' $D/vmstat.log
grep -a -h -iE "Traceback|RuntimeError" $D/*.log | sort | uniq -c | head -5
