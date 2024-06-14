# Prep directory
IPADD="$1"
if [ "$2" == "u" ] || [ "$3" == "u" ]; then
    ssh -i ~/Downloads/fma1.pem ec2-user@"${IPADD}" "mkdir fontmakerai && exit"
    scp -i ~/Downloads/fma1.pem ~/Documents/GitHub/fontmakerai/ml/fontmodel.py ec2-user@"${IPADD}":fontmakerai/
    scp -i ~/Downloads/fma1.pem ~/Documents/GitHub/fontmakerai/ml/runner_runner.py ec2-user@"${IPADD}":fontmakerai/
    scp -i ~/Downloads/fma1.pem ~/Documents/GitHub/config.py ec2-user@"${IPADD}":fontmakerai/
    # scp -i ~/Downloads/fma1.pem ~/Documents/GitHub/fontmakerai/ml/data_no_subr.csv ec2-user@"${IPADD}":fontmakerai/
    scp -i ~/Downloads/fma1.pem ~/Documents/GitHub/fontmakerai/ml/dataset_creator.py ec2-user@"${IPADD}":fontmakerai/
    scp -i ~/Downloads/fma1.pem ~/Documents/GitHub/fontmakerai/ml/tokenizer.py ec2-user@"${IPADD}":fontmakerai/
    # scp -i ~/Downloads/fma1.pem ~/Documents/GitHub/fontmakerai/ml/model.pkl ec2-user@"${IPADD}":fontmakerai/
fi

# Connect
if [ "$2" == "c" ] || [ "$3" == "c" ]; then
    ssh -X -i ~/Downloads/fma1.pem ec2-user@"${IPADD}"
fi