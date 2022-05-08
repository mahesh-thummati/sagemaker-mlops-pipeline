import logging
import tarfile
import os
import sagemaker
import sagemaker.session
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.functions import Join
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.sklearn import SKLearnModel
from sagemaker.workflow.steps import CreateModelStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker import PipelineModel
from sagemaker.model_metrics import MetricsSource, ModelMetrics

from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat,
)

session = sagemaker.session.Session()
region = session.boto_region_name
role = 'arn:aws:iam::539295814914:role/service-role/AmazonSageMaker-ExecutionRole-20220504T213812'
bucket = session.default_bucket()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

class MlOpsPipeline:
    def __init__(self, role, bucket, session):
        self.role = role
        self.bucket = bucket
        self.session = session
        # To what Registry to register the model and its versions.
        self.model_registry_package = ParameterString(name="ModelGroup", default_value="mlops-sklearn-registry")
        # S3 URI to input model
        self.input_model_event = ParameterString(name="InputModelEvent", default_value="")
        # threshold to decide whether or not to register the model with Model Registry
        self.model_performance_threshold = ParameterFloat(name="ModelPerformaceThreshold", default_value=0.7)
        self.model_performace_metric = ParameterString(name="ModelPerformaceMetric", default_value="accuracy")

        # What instance type to use for processing.
        self.processing_instance_type = ParameterString(
            name="ProcessingInstanceType", default_value="ml.m5.large"
        )
        self.sklearn_processor = SKLearnProcessor(
            framework_version="0.23-1", 
            role=role,
            instance_type=self.processing_instance_type, 
            instance_count=1
        )
        # upload local code into s3
        self.prepare_external_model_script_uri = session.upload_data(
            path="src/prepare_external_model.py", 
            key_prefix= "mlops/src/scripts/preprocess"
        )
        self.model_evaluation_script_uri = session.upload_data(
            path="src/evaluate.py", 
            key_prefix= "mlops/src/scripts/evaluate"
        )
        with tarfile.open("src.tar.gz", "w:gz") as tar:
            tar.add("src/", arcname=os.path.basename("src/"))

        self.model_inference_script_uri = session.upload_data(
            path="src.tar.gz",
            key_prefix= "mlops/src/scripts/inference"
        )
    
    def add_prepare_external_model(self):
        """
        Adds step to prepare externally trained model to sagemaker to the ML pipeline
        """
        # Use the sklearn_processor in a SageMaker Pipelines ProcessingStep
        logger.info("Adding model preparation step...")
        step = ProcessingStep(
            name="PrepareExternalModel",
            processor=self.sklearn_processor,
            outputs=[
                ProcessingOutput(
                    output_name="modelpath",
                    source="/opt/ml/processing/output/model/",
                    destination=Join(
                        on="/",
                        values=[
                            f"s3://{self.bucket}/classification",
                            ExecutionVariables.PIPELINE_EXECUTION_ID
                        ],
                    ),
                ),
            ],
            job_arguments=["--bucket", self.bucket, "--key", self.input_model_event],
            code=self.prepare_external_model_script_uri
        )
        return step
    
    def add_evaluate_model(self, model_prep_step):
        """
        Adds evaluation step to perform model testing to the ML pipeline
        """
        # Use the sklearn_processor in a SageMaker Pipelines ProcessingStep
        logger.info("Adding model evaluation step...")
        evaluation_report = PropertyFile(
            name="EvaluationReport", output_name="evaluation", path="evaluation.json"
        )
        step = ProcessingStep(
            name="EvaluateModel",
            processor=self.sklearn_processor,
            inputs=[
                ProcessingInput(
                    source=model_prep_step.properties.ProcessingOutputConfig.Outputs[
                        "modelpath"
                    ].S3Output.S3Uri,
                    destination="/opt/ml/processing/model",
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    source="/opt/ml/processing/evaluation",
                    destination=Join(
                        on="/",
                        values=[
                            f"s3://{bucket}/classification",
                            ExecutionVariables.PIPELINE_EXECUTION_ID,
                            "evaluation-report"
                        ],
                    ),
                ),
            ],
            code=self.model_evaluation_script_uri,
            property_files=[evaluation_report]
        )
        return step, evaluation_report
    
    def register_pipeline_model(self, model_prep_step, evaluation_step):
        """
        Adds model registration step to register an approved model to the ML pipeline
        """
        logger.info("Adding register pipeline model step...")
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=Join(
                    on="/",
                    values=[
                        evaluation_step.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"],
                        "evaluation.json",
                    ],
                ),
                content_type="application/json",
            )
        )
        sklearn_model = SKLearnModel(
            name="SKLearnClassificationModel",
            model_data=Join(
                on="/",
                values=[
                    model_prep_step.properties.ProcessingOutputConfig.Outputs["modelpath"].S3Output.S3Uri,
                    "model.tar.gz"
                ]
            ),
            role=self.role,
            sagemaker_session=self.session,
            entry_point="src/inference.py",
            source_dir=self.model_inference_script_uri,
            framework_version="0.23-1",
            #image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
            py_version='py3')
        

        pipeline_model = PipelineModel(models=[sklearn_model],role=self.role, sagemaker_session=self.session)

        step_register = RegisterModel(
            name="SKLearnClassificationRegisterModel",
            model=pipeline_model,
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
            transform_instances=["ml.m5.xlarge"],
            model_package_group_name=self.model_registry_package,
            model_metrics=model_metrics
        )
        return step_register
    
    def create_pipeline(self, name="MlOpsSamplePipeline"):
        """
        Creates SageMaker ML Pipeline
        """
        prepare_external_model_step = self.add_prepare_external_model()
        evaluation_step, evaluation_report = self.add_evaluate_model(prepare_external_model_step)
        register_model_step = self.register_pipeline_model(prepare_external_model_step, evaluation_step)

        cond_gte = ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=evaluation_step.name,
                property_file=evaluation_report,
                json_path="binary_classification_metrics.accuracy.value",
            ),
            right=self.model_performance_threshold,
        )
        step_cond = ConditionStep(
            name="Accuracy-Condition",
            conditions=[cond_gte],
            if_steps=[register_model_step],
            else_steps=[],
        )

        pipeline = Pipeline(
            name=name,
            parameters=[
                self.model_registry_package,
                self.input_model_event,
                self.model_performace_metric,
                self.model_performance_threshold,
                self.processing_instance_type
            ],
            steps=[prepare_external_model_step, evaluation_step, step_cond]
        )
        return pipeline
    
    def upsert_pipeline(self, pipeline):
        pipeline.upsert(role_arn=self.role)
    

if __name__ == "__main__":
    logger.info("Preparing to create ML pipeline...")
    mlops = MlOpsPipeline(role, bucket, session)
    pipeline = mlops.create_pipeline("MLOpsPipeline")
    mlops.upsert_pipeline(pipeline)
    logger.info(f"ML pipeline {pipeline.name} created successfully...")