#!/bin/bash
source .env
echo "Generating JWT key..."
echo "Password: ${JWT_KEY_FONT_API_PASSWORD}"
echo "Command: ssh-keygen -q -t rsa -N ${JWT_KEY_FONT_API_PASSWORD} -f ./id_rsa_font_api"
echo "Output:"
ssh-keygen -q -t rsa -N ${JWT_KEY_FONT_API_PASSWORD} -f ./id_rsa_font_api