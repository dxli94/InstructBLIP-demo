TAG="gcr.io/salesforce-research-internal/instructblip-demo"
gcloud builds submit . -t=$TAG --machine-type=n1-highcpu-32 --timeout=9000
