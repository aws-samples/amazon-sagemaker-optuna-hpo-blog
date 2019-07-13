import base64
content = base64.b64encode(b"""#!/bin/bash
sudo -u ec2-user -i <<'EOF'
cd SageMaker && git clone https://github.com/aws-samples/aws-sagemaker-optuna-hpo-blog.git
EOF""")
print(content)
