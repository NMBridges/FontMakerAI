# Prep directory
IPADD="$1"
KEY=~/Downloads/elo.pem
FONTMODEL=~/Documents/GitHub/fontmakerai/ml/fontmodel.py
RUNRUN=~/Documents/GitHub/fontmakerai/ml/runner_runner.py
CONF=~/Documents/GitHub/fontmakerai/config.py
DSETC=~/Documents/GitHub/fontmakerai/ml/dataset_creator.py
TOK=~/Documents/GitHub/fontmakerai/ml/tokenizer.py
DATA9000=~/Documents/GitHub/fontmakerai/ml/expanded_ninethousand.csv
VIZ=~/Documents/GitHub/fontmakerai/parsing/glyph_viz.py
if [ "$2" == "u" ] || [ "$3" == "u" ] || [ "$4" == "u" ]; then
    ssh -i $KEY ec2-user@${IPADD} "mkdir fontmakerai && exit"
    ssh -i $KEY ec2-user@"${IPADD}" "cd fontmakerai && mkdir training_images && exit"
    scp -i $KEY $FONTMODEL $RUNRUN $CONF $DSETC $TOK $VIZ ec2-user@"${IPADD}":fontmakerai/
fi

# Connect
if [ "$2" == "c" ] || [ "$3" == "c" ] || [ "$4" == "c" ]; then
    ssh -X -i $KEY ec2-user@${IPADD}
fi

# Download training images
if [ "$2" == "i" ] || [ "$3" == "i" ] || [ "$4" == "i" ]; then
    scp -i $KEY -r ec2-user@${IPADD}:~/fontmakerai/training_images/ .
fi