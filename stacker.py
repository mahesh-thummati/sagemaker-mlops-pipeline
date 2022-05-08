from time import sleep, time
from urllib import response
import boto3
from botocore.exceptions import ClientError

def does_stack_exist(stack_name):
    """
    Verifies if cloud formation stack exists
    """
    client = boto3.client('cloudformation')
    try:
        response = client.describe_stacks(StackName=stack_name)
        return response["Stacks"][0]["StackName"]
    except ClientError as exp:
        return None
    
def upsert_cf_stack(stack_name, template_body, params_list=[{}]):
    """
    Creates/Updates cloud formation stack
    """
    client = boto3.client('cloudformation')
    desc_stack_name = does_stack_exist(stack_name)
    if desc_stack_name: # stack exists
        print(f"Stack -- {stack_name} already exists. Attempting to update stack...")
        try:
            response = client.update_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=params_list,
                Capabilities=["CAPABILITY_NAMED_IAM"]
            )
            return does_stack_exist(stack_name)
        except ClientError as exp:
            print(exp)
    else:
        print(f"Stack -- {stack_name} doesn't exist. Attempting to create stack...")
        response = client.create_stack(
            StackName=stack_name,
            TemplateBody=template_body,
            Parameters=params_list,
            Capabilities=["CAPABILITY_NAMED_IAM"]
        )
        return does_stack_exist(stack_name)
    # try creating the stack first, if already exists update stack
    return desc_stack_name

def monitor_cf_stack(stack_name, timeout_s=300):
    """
    Monitors cloudformation stack
    """
    start_time = time()
    client = boto3.client('cloudformation')
    response = client.describe_stacks(StackName=stack_name)["Stacks"][0]
    while ("IN_PROGRESS" in response["StackStatus"]):
        # check if timeout happended
        if time() - start_time > timeout_s:
            return "TimeOut"
        print(f"Describing stack {stack_name}...")
        response = client.describe_stacks(StackName=stack_name)["Stacks"][0]
        sleep(10)
    return response["StackStatus"]

def retrieve_stack_outputs(stack_name):
    """
    Returns stack output variables as dict
    """
    client = boto3.client('cloudformation')
    response = client.describe_stacks(StackName=stack_name)["Stacks"][0]
    cf_output_vals = {}
    for stack_out in response['Outputs']:
        cf_output_vals[stack_out["OutputKey"]] = stack_out["OutputValue"]
    return cf_output_vals
