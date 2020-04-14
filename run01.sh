mkdir -p data_wav

for f in `ls data/*.als`
do
b=`basename $f .als`
#mp4alsRM23/bin/linux/mp4alsRM23	-x $f data_wav/${b}.wav
done

mkdir -p data_wav1ch
for f in `ls data_wav/*.wav`
do
b=`basename $f`
sox data_wav/${b} data_wav1ch/${b} remix 1
done


