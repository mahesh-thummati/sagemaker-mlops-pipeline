# sagemaker-mlops-pipeline
Hosts AWS SageMaker ML OPS pipeline and related infrastructure.
This repo contains code to accept externally trained scikit learn model, perform model testing and register the model in sagemaker model repository 
based on the acceptable levels of performance metrics.

## Resources created
1. An S3 bucket (default: sagemaker-mlops-{region}-{accountid}) and enable event bridge notification
2. An IAM role (EventBridgeMLPipelineTriggerRole-{region}) for event bridge rule to trigger sagemaker ML pipeline
3. An event bridge rule to trigger ML pipeline based on create object notification from the S3 bucket
4. An IAM role (SageMakerMLOpsExecutionRole-{region}) for SageMaker ML Pipeline to assume
5. SageMaker ML Pipeline (MLOpsPipeline)

## Steps to create infrastructure
Run deploy.py program which will create following resources:
1. cloudformation stack to create sagemaker IAM role and s3 bucket
2. python code to create ML pipeline
3. cloudformation template to create event bridge IAM role and event bridge rule

## Steps to test the pipeline
Upload externally trained scikit learn classification model to sagemaker-mlops-{region}-{accountid}/input/ and monitor the status of pipeline in sagemaker studio

