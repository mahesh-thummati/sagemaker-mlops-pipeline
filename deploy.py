import os
import sys
import json
from re import template
import sagemaker
import sagemaker.session

import stacker
from ml_ops_pipeline import MlOpsPipeline


def load_template_file(cf_template):
    template_file = os.path.abspath(os.path.join(os.path.abspath(__file__), "../", "templates", cf_template))
    with open(template_file, "r") as f:
        template_body=json.load(f)
    #print(template_body)
    return json.dumps(template_body)

if __name__ == "__main__":
    # create IAM and s3 bucket
    s3_iam_stack = stacker.upsert_cf_stack(
        "SageMakerMLOpsIAMAndS3",
        load_template_file("iam-s3.json")
    )
    status = stacker.monitor_cf_stack(s3_iam_stack, 3600)
    print (f"stack -- {s3_iam_stack} --finished with status -- {status} --")
    if status != "CREATE_COMPLETE" and status != "UPDATE_COMPLETE":
        print ("Failure occurred in stack creation.. exitting the process")
        sys.exit(1)

    s3_iam_output = stacker.retrieve_stack_outputs(s3_iam_stack)
    print(s3_iam_output)
    
    #create ML pipeline
    session = sagemaker.session.Session()
    role = s3_iam_output["SageMakerMLPipelineExecutionRole"]
    bucket = s3_iam_output["SageMakerS3Bucket"]
    mlops = MlOpsPipeline(role, bucket, session)
    pipeline = mlops.create_pipeline("MLOpsPipeline")
    mlops.upsert_pipeline(pipeline)
    print(f"ML pipeline {pipeline.name} created successfully...")

    # create event bridge rule
    params = [
        {
            "ParameterKey": "SageMakerMLPipelineName",
            "ParameterValue": pipeline.name.lower()
        },
        {
            "ParameterKey": "SageMakerS3Bucket",
            "ParameterValue": bucket
        }
    ]
    event_bridge_stack = stacker.upsert_cf_stack(
        "MLOpsEventBridgeStack",
        load_template_file("eventbridge-iam.json"),
        params
    )
    status = stacker.monitor_cf_stack(event_bridge_stack, 3600)
    print (f"stack -- {event_bridge_stack} --finished with status -- {status} --")
    if status != "CREATE_COMPLETE" and status != "UPDATE_COMPLETE":
        print ("Failure occurred in stack creation.. exitting the process")
        sys.exit(1)
    
    print("Deployment of ML OPS infrastructure is successful!")
    print(f"Upload your model to s3://{bucket}/input/ to trigger ML OPS Process!")




