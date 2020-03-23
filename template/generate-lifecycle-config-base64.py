import base64
content = base64.b64encode(b"""#!/bin/bash
sudo -u ec2-user -i <<'EOF'
cd /home/ec2-user/SageMaker && git clone https://github.com/aws-samples/amazon-sagemaker-optuna-hpo-blog.git
EOF""")
print(content)
