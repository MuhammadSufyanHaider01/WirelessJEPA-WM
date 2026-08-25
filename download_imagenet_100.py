import kagglehub

# Download latest version
path = kagglehub.dataset_download("ambityga/imagenet100")

print("Path to dataset files:", path)