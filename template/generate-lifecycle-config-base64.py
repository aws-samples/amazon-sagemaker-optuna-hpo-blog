import base64
content = base64.b64encode(b"""#!/bin/bash
set -e
cd SageMaker && git clone https://github.com/aws-samples/aws-sagemaker-optuna-hpo-blog.git""")
print(content)
