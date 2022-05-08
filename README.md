# sagemaker-mlops-pipeline
Hosts AWS SageMaker ML OPS pipeline.
This repo contains code to accept externally trained scikit learn model, perform model testing and register the model in sagemaker model repository 
based on the acceptable levels of performance metrics.

## Resources created
1. An S3 bucket (default: sagemaker_{region}-{accountid}) and enable event bridge notification
2. An IAM role for event bridge rule to trigger sagemaker ML pipeline
3. An event bridge rule to trigger ML pipeline based on create object notification from the S3 bucket
4. An IAM role for SageMaker ML Pipeline to assume
5. SageMaker ML Pipeline

## Steps to create infrastructure
1. run cloudformation template to create sagemaker IAM role and s3 bucket
2. run python code to create ML pipeline
3. run cloudformation template to create event bridge IAM role and event bridge rule

## Steps to test the pipeline
Upload externally trained scikit learn classification model to sagemaker_{region}-{accountid}/input/ and monitor the status of pipeline in sagemaker studio

