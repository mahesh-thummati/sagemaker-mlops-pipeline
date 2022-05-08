import tarfile
import argparse
import boto3
import json
import sys
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def make_tarfile(output_filename, inp_file):
    """
    Converts model.joblib to model.tar.gz
    """
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(inp_file, arcname=os.path.basename(inp_file))
    
def create_local_dir(file_path:str)->bool:
    """
    creates local directory if does not exist
    """
    try:
        directory = os.path.dirname(file_path)
        if directory == "":
            return True # nothing to create
        if not os.path.exists(directory):
            os.makedirs(directory)
    except Exception as exc:
        print("Error {} occurred while creating local directory".format(exc))
        return False
    
    return True
def download_file(bucket, key, local_dest):
    """
    copies an s3 file to local
    """
    logger.info(f"Downloading {key} to local...")
    s3_resource = boto3.resource('s3')
    try:
        create_local_dir(local_dest)
        s3_resource.Bucket(bucket).download_file(key, local_dest)
    except Exception as exc:
        print ("Error {} occurred while working on s3 object to local.".format(exc))
        return False
    return True

def upload_file(src: str, dest_bucket: str, dest_key: str)->bool:
    """
    copies local file to s3
    """
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(src, dest_bucket, dest_key)
    except Exception as exc:
        print("Error {} occurred while working on local object to s3.".format(exc))
        return False
    
    return True

if __name__ == "__main__":
    input_model = "/opt/ml/processing/input/model/model.joblib"
    output_model = "/opt/ml/processing/output/model/model.tar.gz"
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, dest='bucket')
    parser.add_argument('--key', type=str, dest='key')
    logger.info("Parsing input arguments to processing job")
    
    args = parser.parse_args()
    #logger.info("input argument", str(args.event_obj))
    #inp_dict = json.loads(args.nucket)
    logger.info("input parameters", args)
    bucket = args.bucket
    key = args.key
    if not download_file(bucket, key, input_model):
        sys.exit(1)
    make_tarfile(output_model, input_model)
    #upload_file(output_model, bucket, "processing")