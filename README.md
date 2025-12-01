# Face Recognition Script

A Python script for performing face recognition using the `deepface` library. This script finds the most similar faces in a local image database when given a target image.

## How it Works

The script leverages state-of-the-art face recognition models to generate embeddings for face images. It takes a specified target image and compares its face embedding against the embeddings of all images in a designated database folder (`Recon_Pool`).

The script then outputs a sorted list of the most similar faces found in the database, including the distance metric (e.g., 'cosine') which indicates the similarity. A lower distance means a higher likelihood of a match.

## Technologies Used

*   **Python**
*   **DeepFace**: A lightweight face recognition and facial attribute analysis library for Python.
*   **Pandas**: For data manipulation and displaying results in a structured format.

## Setup and Installation

1.  **Clone the repository:**
    ```
    git clone https://github.com/GeorgeMrls/Face_Recon.git
    cd ./Face_Recon
    ```

2.  **Create and activate a virtual environment:**
    ```
    python -m venv venv_deepface
    source venv_deepface/bin/activate
    ```

3.  **Install the required dependencies:**

    ```
    pip install deepface tf-keras # opencv and pandas will automatically added due to deepface dependencies
    ```

## Configuration and Usage

1.  **Prepare your images:**
    *   Place the images you want to search through in the `Recon_Pool` directory.
    *   Place the image of the person you want to find in the `targets` directory.

2.  **Configure the script:**
    Open `Face_Recon.py` and modify the following variables:
    *   `DB_PATH`: Path to your image database (defaults to `"Recon_Pool"`).
    *   `TARGET_IMG`: Path to your target image (e.g., `"targets/image.jpg"`).
    *   `MODEL_NAME`: The face recognition model to use. `ArcFace` is recommended for high accuracy.

3.  **Run the script:**
    ```
    python Face_Recon.py
    ```

## Example Output

After running the script, you will see a pandas DataFrame containing the top matches, ranked by similarity.

```
23:56:13 - Searching targets/Aaron_Peirsol_0001.jpg in 16 length datastore
23:56:14 - find function duration 2.696988821029663 seconds
                            identity                                      hash  target_x  target_y  ...  source_w  source_h  distance  confidence
0  Recon_Pool/Aaron_Peirsol_0001.jpg  6fc852d883de3d5ea73e15b2c83b5555cb6014f0        78        62  ...        96       128  0.040297      100.00
1  Recon_Pool/Aaron_Peirsol_0002.jpg  463ee7af05a23be76e855a758c86c9cf602719e9        75        61  ...        98       133  0.299847       84.74
2  Recon_Pool/Aaron_Peirsol_0003.jpg  ceb8d2ae8b87ba0fc0a7c75c02a58cbd0a76b038        76        59  ...        95       134  0.339493       80.21
3  Recon_Pool/Aaron_Peirsol_0004.jpg  dcaccadfb44fc1ceb2bb75d53434a54cdc8ee08b        84        61  ...        90       133  0.541728       55.99

[4 rows x 13 columns]
```

## Dataset

This project uses sample images from the [Labeled Faces in the Wild (LFW)](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) dataset for demonstration. The `targets` and `Recon_Pool` directories contain images from this dataset. The full LFW dataset is not included in this repository due to its large size.