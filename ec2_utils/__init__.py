from .config import *

# SSH_KEY = get_ssh_key(KEY_NAME)
SG_ID = get_security_group(
    sg_name=SECURITY_GROUP['name'],
    sg_desc=SECURITY_GROUP['desc'],
    vpc_id=None
)

