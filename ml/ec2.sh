# Prep directory
IPADD="$1"
KEY=~/Downloads/elo.pem
FONTMODEL=~/Documents/GitHub/fontmakerai/ml/fontmodel.py
RUNRUN=~/Documents/GitHub/fontmakerai/ml/runner_runner.py
CONF=~/Documents/GitHub/fontmakerai/config.py
CONFTXT=~/Documents/GitHub/fontmakerai/.config.txt
DSETC=~/Documents/GitHub/fontmakerai/ml/dataset_creator.py
TOK=~/Documents/GitHub/fontmakerai/ml/tokenizer.py
DATA47000=~/Documents/GitHub/fontmakerai/ml/47000_fonts.csv
VIZ=~/Documents/GitHub/fontmakerai/parsing/glyph_viz.py
PERF=~/Documents/GitHub/fontmakerai/ml/performance.py
TLUTILS=~/Documents/GitHub/fontmakerai/parsing/tablelist_utils.py
SCLDS=~/Documents/GitHub/fontmakerai/ml/scale_dataset.py

if [ "$2" == "u" ] || [ "$3" == "u" ] || [ "$4" == "u" ]; then
    ssh -i $KEY ec2-user@${IPADD} "mkdir fontmakerai && exit"
    ssh -i $KEY ec2-user@"${IPADD}" "cd fontmakerai && mkdir training_images && exit"
    scp -i $KEY $FONTMODEL $RUNRUN $CONF $CONFTXT $DSETC $TOK $VIZ $PERF $TLUTILS $SCLDS ec2-user@"${IPADD}":fontmakerai/
fi

# Connect
if [ "$2" == "c" ] || [ "$3" == "c" ] || [ "$4" == "c" ]; then
    ssh -X -i $KEY ec2-user@${IPADD}
fi

# Download training images
if [ "$2" == "i" ] || [ "$3" == "i" ] || [ "$4" == "i" ]; then
    scp -i $KEY -r ec2-user@${IPADD}:~/fontmakerai/training_images/*.png training_images/
    #scp -i $KEY -r ec2-user@${IPADD}:~/fontmakerai/training_images/*.txt .
fi

# Download output images
if [ "$2" == "o" ] || [ "$3" == "o" ] || [ "$4" == "o" ]; then
    scp -i $KEY -r ec2-user@${IPADD}:~/fontmakerai/training_images/out.png training_images/out.png
fi

IMDAGE=~/Documents/GitHub/fontmakerai/ml/image_dataset_generator.py
DIFFMOD=~/Documents/GitHub/fontmakerai/ml/diffusion_model.py
DIFFRUN=~/Documents/GitHub/fontmakerai/ml/diffusion_runner.py
UNET=~/Documents/GitHub/fontmakerai/ml/unet.py

if [ "$2" == "d" ] || [ "$3" == "d" ] || [ "$4" == "d" ]; then
    scp -i $KEY $IMDAGE $DIFFMOD $DIFFRUN $UNET ec2-user@"${IPADD}":fontmakerai/
fi