import gdown, zipfile, os

def download_model():
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Google Drive link (your file)
    file_id = "1wPump7hM0JDtUk7Gvz0Q_AwLIIYf2Kgv"  # your file ID
    url = f"https://drive.google.com/uc?id={file_id}"

    output_zip = os.path.join(model_dir, "model_folder.zip")

    # download
    print("Downloading model...")
    gdown.download(url, output_zip, quiet=False)

    # unzip
    print("Extracting model files...")
    with zipfile.ZipFile(output_zip, "r") as zip_ref:
        zip_ref.extractall(model_dir)

    print("Model downloaded and extracted successfully.")
