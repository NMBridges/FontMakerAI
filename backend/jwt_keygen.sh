#!/bin/bash
source .env
ssh-keygen -q -t rsa -N ${JWT_KEY_FONT_API_PASSWORD} -f ./id_rsa_font_api