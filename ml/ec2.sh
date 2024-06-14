# Prep directory
IPADD="$1"
FONTMODEL=~/Documents/GitHub/fontmakerai/ml/fontmodel.py
RUNRUN=~/Documents/GitHub/fontmakerai/ml/runner_runner.py
CONF=~/Documents/GitHub/fontmakerai/config.py
DSETC=~/Documents/GitHub/fontmakerai/ml/dataset_creator.py
TOK=~/Documents/GitHub/fontmakerai/ml/tokenizer.py
VIZ=~/Documents/GitHub/fontmakerai/parsing/glyph_viz.py
if [ "$2" == "u" ] || [ "$3" == "u" ]; then
    ssh -i ~/Downloads/fma1.pem ec2-user@"${IPADD}" "mkdir fontmakerai && exit"
    ssh -i ~/Downloads/fma1.pem ec2-user@"${IPADD}" "cd fontmakerai && mkdir training_images && exit"
    scp -i ~/Downloads/fma1.pem $FONTMODEL $RUNRUN $CONF $DSETC $TOK $VIZ ec2-user@"${IPADD}":fontmakerai/
    # scp -i ~/Downloads/fma1.pem ~/Documents/GitHub/fontmakerai/ml/data_no_subr.csv ec2-user@"${IPADD}":fontmakerai/
    # scp -i ~/Downloads/fma1.pem ~/Documents/GitHub/fontmakerai/ml/model.pkl ec2-user@"${IPADD}":fontmakerai/
fi

# Connect
if [ "$2" == "c" ] || [ "$3" == "c" ]; then
    ssh -X -i ~/Downloads/fma1.pem ec2-user@"${IPADD}"
fi