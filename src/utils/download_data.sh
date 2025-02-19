#!/bin/bash

DATA_URL="https://drive.google.com/drive/folders/1V0UEf9TG8c7UmzBOIboh-UUoiQ0ZUZox?usp=drive_link"
MODEL_URL="https://drive.google.com/drive/folders/1ESgZgkqz9okM9XJnmfFGXaoz3QMnzYI2?usp=drive_link"

echo "Downloading folder 'data'..."
gdown --folder "$DATA_URL"

echo "Downloading folder 'model'..."
gdown --folder "$MODEL_URL"

echo "Download completed."
