from aws_cdk import Stack, App, Fn
from constructs import Construct

from aws_cdk import aws_lambda as _lambda
from aws_cdk import aws_cloudfront as cloudfront
from aws_cdk import aws_cloudfront_origins as origins
from aws_cdk import aws_certificatemanager as acm
from aws_cdk import aws_ecr_assets as ecr_assets

domain_name = "laptop-price-prediction.tzvi.dev"

class LambdaWithCloudFrontStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Create the Lambda function
        lambda_function = _lambda.DockerImageFunction(
            self, id="LaptopPredictionFunction",
            code= _lambda.DockerImageCode.from_image_asset(directory=".",file="Dockerfile-lambda", platform=ecr_assets.Platform.LINUX_ARM64),
            architecture= _lambda.Architecture.ARM_64,
        )

        # Create a Function URL for the Lambda
        function_url = lambda_function.add_function_url(
            auth_type=_lambda.FunctionUrlAuthType.NONE  # Or configure as needed
        )

        certificate = acm.Certificate(self,
                                      "LaptopPredictionCertificate", 
                                      domain_name=domain_name, 
                                      validation=acm.CertificateValidation.from_email(),
                                      )
        
        # Create a CloudFront Distribution
        cloudfront_distribution = cloudfront.Distribution(
            self, "LaptopPredictionDistribution",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.HttpOrigin(
                    # https://github.com/aws/aws-cdk/issues/20254
                    domain_name=Fn.select(2, Fn.split('/', function_url.url)),  # Use the function URL as the origin
                    origin_path="/",  # You can adjust this as needed
                    
                ),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
            ),
            certificate=certificate,
            domain_names=[domain_name]
        )

app = App()
LambdaWithCloudFrontStack(app, "LaptopPredictionStack")
app.synth()