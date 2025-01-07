# Prep directory
IPADD="$1"
# USR="ubuntu"
USR="ec2-user"
KEY=~/Downloads/elo.pem
FONTMODEL=~/Documents/GitHub/fontmakerai/ml/fontmodel.py
# RUNRUN=~/Documents/GitHub/fontmakerai/ml/runner_runner.py
CFFRUN=~/Documents/GitHub/fontmakerai/ml/train-cff.py
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
FLTR=~/Documents/GitHub/fontmakerai/parsing/filter_allchars.py
CSVTOBASIC=~/Documents/GitHub/fontmakerai/dataset_utils/raw_csv_to_basic_csv.py
CSVTOPT=~/Documents/GitHub/fontmakerai/dataset_utils/csv_to_pt.py
OPVAE=~/Documents/GitHub/fontmakerai/ml/op_vae.py


if [ "$2" == "u" ] || [ "$3" == "u" ] || [ "$4" == "u" ]; then
    jupyter nbconvert ~/Documents/GitHub/fontmakerai/ml/train-cff.ipynb --to script
    mv ~/Documents/GitHub/fontmakerai/ml/train-cff.txt ~/Documents/GitHub/fontmakerai/ml/train-cff.py
    # ssh -i $KEY $USR@${IPADD} "mkdir fontmakerai && exit"
    # ssh -i $KEY $USR@"${IPADD}" "cd fontmakerai && mkdir training_images && exit"
    scp -i $KEY $FONTMODEL $CFFRUN $CONF $CONFTXT $DSETC $TOK $VIZ $PERF $TLUTILS $SCLDS $FLTR $CSVTOBASIC $CSVTOPT $OPVAE $USR@"${IPADD}":fontmakerai/
fi

# Download training images
if [ "$2" == "i" ] || [ "$3" == "i" ] || [ "$4" == "i" ]; then
    scp -i $KEY -r $USR@${IPADD}:~/fontmakerai/training_images/*.png training_images/
    #scp -i $KEY -r ec2-user@${IPADD}:~/fontmakerai/training_images/*.txt .
fi

# Download output images
if [ "$2" == "o" ] || [ "$3" == "o" ] || [ "$4" == "o" ]; then
    scp -i $KEY -r $USR@${IPADD}:~/fontmakerai/training_images/samples/ training_images/
fi

IMDAGE=~/Documents/GitHub/fontmakerai/dataset_utils/gen_all_chars.py
DIFFRUN=~/Documents/GitHub/fontmakerai/ml/train-diffusion.py
UNET=~/Documents/GitHub/fontmakerai/ml/backbones/unet.py
ATTN=~/Documents/GitHub/fontmakerai/ml/backbones/attention.py
VAE=~/Documents/GitHub/fontmakerai/ml/vae.py
LDM=~/Documents/GitHub/fontmakerai/ml/ldm.py
DDPM=~/Documents/GitHub/fontmakerai/ml/ddpm.py

if [ "$2" == "d" ] || [ "$3" == "d" ] || [ "$4" == "d" ]; then
    jupyter nbconvert ~/Documents/GitHub/fontmakerai/ml/train-diffusion.ipynb --to script
    mv ~/Documents/GitHub/fontmakerai/ml/train-diffusion.txt ~/Documents/GitHub/fontmakerai/ml/train-diffusion.py
    scp -i $KEY $IMDAGE $DDPM $LDM $DIFFRUN $UNET $ATTN $VAE $USR@"${IPADD}":fontmakerai/
fi

# Connect
if [ "$2" == "c" ] || [ "$3" == "c" ] || [ "$4" == "c" ]; then
    ssh -X -i $KEY $USR@${IPADD}
fi