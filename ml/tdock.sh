# Prep directory
IPADD="$1"
KEY=~/Documents/tdock.pem
FONTMODEL=~/Documents/GitHub/fontmakerai/ml/fontmodel.py
RUNRUN=~/Documents/GitHub/fontmakerai/ml/runner_runner.py
CONF=~/Documents/GitHub/fontmakerai/config.py
CONFTXT=~/Documents/GitHub/fontmakerai/.config.txt
DSETC=~/Documents/GitHub/fontmakerai/dataset_utils/dataset_loader.py
TOK=~/Documents/GitHub/fontmakerai/ml/tokenizer.py
DATA47000=~/Documents/GitHub/fontmakerai/ml/47000_fonts.csv
DATA1900K=~/Documents/GitHub/fontmakerai/ml/1900k.csv
VIZ=~/Documents/GitHub/fontmakerai/parsing/glyph_viz.py
PERF=~/Documents/GitHub/fontmakerai/ml/performance.py
TLUTILS=~/Documents/GitHub/fontmakerai/parsing/tablelist_utils.py
SCLDS=~/Documents/GitHub/fontmakerai/dataset_utils/scale_dataset.py

if [ "$2" == "u" ] || [ "$3" == "u" ] || [ "$4" == "u" ]; then
    sshpass -f"${KEY}" ssh -p 10001 user@${IPADD} "mkdir fontmakerai && exit"
    sshpass -f"${KEY}" ssh -p 10001 user@"${IPADD}" "cd fontmakerai && mkdir training_images && exit"
    sshpass -f"${KEY}" scp -P 10001 -v $FONTMODEL $RUNRUN $CONF $CONFTXT $DSETC $TOK $VIZ $PERF $TLUTILS $SCLDS user@"${IPADD}":fontmakerai/
fi

# Connect
if [ "$2" == "c" ] || [ "$3" == "c" ] || [ "$4" == "c" ]; then
    sshpass -f"${KEY}" ssh -p 10001 -X user@${IPADD}
fi

# Download training images
if [ "$2" == "i" ] || [ "$3" == "i" ] || [ "$4" == "i" ]; then
    sshpass -f"${KEY}" scp -P 10001 -r user@${IPADD}:~/fontmakerai/training_images/*.png training_images/
    #scp -i $KEY -r ec2-user@${IPADD}:~/fontmakerai/training_images/*.txt .
fi

# Download output images
if [ "$2" == "o" ] || [ "$3" == "o" ] || [ "$4" == "o" ]; then
    sshpass -f"${KEY}" scp -P 10001 -r user@${IPADD}:~/fontmakerai/training_images/samples/ training_images/
fi

IMDAGE=~/Documents/GitHub/fontmakerai/dataset_utils/gen_all_chars.py
DIFFMOD=~/Documents/GitHub/fontmakerai/ml/diffusion_model.py
DIFFRUN=~/Documents/GitHub/fontmakerai/ml/diffusion_runner.py
UNET=~/Documents/GitHub/fontmakerai/ml/unet.py

if [ "$2" == "d" ] || [ "$3" == "d" ] || [ "$4" == "d" ]; then
    sshpass -f"${KEY}" scp -P 10001 $IMDAGE $DIFFMOD $DIFFRUN $UNET user@"${IPADD}":fontmakerai/
fi