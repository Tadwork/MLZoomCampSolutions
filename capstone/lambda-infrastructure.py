from aws_cdk import Stack, App, Fn, Duration
from constructs import Construct

from aws_cdk import aws_lambda as _lambda
from aws_cdk import aws_cloudfront as cloudfront
from aws_cdk import aws_cloudfront_origins as origins
from aws_cdk import aws_certificatemanager as acm
from aws_cdk import aws_ecr_assets as ecr_assets

domain_name = "face-age-detection.tzvi.dev"

lambda_memory_size = 768
lambda_timeout = Duration.seconds(10)


class LambdaWithCloudFrontStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Create the Lambda function
        lambda_function = _lambda.DockerImageFunction(
            self,
            id="FaceAgeDetectionFunction",
            code=_lambda.DockerImageCode.from_image_asset(
                directory=".",
                file="Dockerfile-lambda",
                platform=ecr_assets.Platform.LINUX_AMD64,
            ),
            architecture=_lambda.Architecture.X86_64,
            memory_size=lambda_memory_size,
            timeout=lambda_timeout,
        )

        # Create a Function URL for the Lambda
        function_url = lambda_function.add_function_url(
            auth_type=_lambda.FunctionUrlAuthType.NONE  # Or configure as needed
        )

        certificate = acm.Certificate(
            self,
            "FaceAgeDetectionCertificate",
            domain_name=domain_name,
            validation=acm.CertificateValidation.from_email(),
        )

        # Create a CloudFront Distribution
        cloudfront_distribution = cloudfront.Distribution(
            self,
            "FaceAgeDetectionDistribution",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.HttpOrigin(
                    # https://github.com/aws/aws-cdk/issues/20254
                    domain_name=Fn.select(
                        2, Fn.split("/", function_url.url)
                    ),  # Use the function URL as the origin
                    origin_path="/",  # You can adjust this as needed
                ),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.ALLOW_ALL,
                allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
            ),
            certificate=certificate,
            domain_names=[domain_name],
        )


app = App()
LambdaWithCloudFrontStack(app, "FaceAgeDetectionStack")
app.synth()
