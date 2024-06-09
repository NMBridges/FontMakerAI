# Prep directory
IPADD="$1"
ssh -i ~/Downloads/fma1.pem ec2-user@"${IPADD}" "mkdir fontmakerai && exit"
scp -i ~/Downloads/fma1.pem ~/Documents/GitHub/fontmakerai/fontmodel.py ec2-user@"${IPADD}":fontmakerai/
scp -i ~/Downloads/fma1.pem ~/Documents/GitHub/fontmakerai/fparser.py ec2-user@"${IPADD}":fontmakerai/
# scp -i ~/Downloads/fma1.pem ~/Documents/GitHub/fontmakerai/model.pkl ec2-user@"${IPADD}":fontmakerai/
scp -i ~/Downloads/fma1.pem ~/Documents/GitHub/fontmakerai/runner_runner.py ec2-user@"${IPADD}":fontmakerai/

# Connect
ssh -X -i ~/Downloads/fma1.pem ec2-user@"${IPADD}"