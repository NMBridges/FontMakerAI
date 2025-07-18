import dotenv
from cryptography.hazmat.primitives import serialization
import torch
from enum import Enum

class FontRunStage(Enum):
    EMPTY_DESCRIPTION = 0
    IMAGES_GENERATING = 1
    IMAGES_GENERATED = 2
    VECTORIZATION_STAGE = 3
    DOWNLOAD_STAGE = 4


NUM_GLYPHS = 26


# Load JWT key
password = dotenv.get_key('.env', 'JWT_KEY_FONT_API_PASSWORD')
private_key = open('/id_rsa', 'r').read()
public_key = open('/id_rsa.pub', 'r').read()
jwt_public_key = serialization.load_ssh_public_key(public_key.encode())
jwt_private_key = serialization.load_ssh_private_key(private_key.encode(), password=password.encode())
assert jwt_private_key is not None, "JWT private key is not set"
assert jwt_public_key is not None, "JWT public key is not set"


device = 'cuda'
threads = {}

LOCAL_DEBUG = False
LOAD_MODELS = not LOCAL_DEBUG


diff_dtype = torch.float32
path_dtype = torch.float16
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True