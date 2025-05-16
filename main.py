import os
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import shutil

def get_jpeg_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.jpeg', '.jpg'))]

def _calculate_channel_histogram(image_array_flattened, channel_index, bins):
    return np.histogram(image_array_flattened[channel_index::3], bins=bins)[0]

def extract_color_histogram(image_path, bins=8):
    try:
        img = Image.open(image_path).convert('RGB')
        img_array_flattened = np.array(img).flatten()
        histograms = []
        for i in range(3):
            histogram_channel = _calculate_channel_histogram(img_array_flattened, i, bins)
            histograms.append(histogram_channel)
        histogram = np.concatenate(histograms)
        return histogram / (img.width * img.height) # 正規化
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def group_similar_images_by_color_and_output(input_directory, output_directory, n_clusters):
    image_files = get_jpeg_files(input_directory)
    features = []
    valid_files = []
    for file in image_files:
        histogram = extract_color_histogram(file)
        if histogram is not None:
            features.append(histogram)
            valid_files.append(file)

    if not features:
        print("No valid JPEG images found.")
        return

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(features)
    labels = kmeans.labels_

    grouped_images = defaultdict(list)
    for i, label in enumerate(labels):
        grouped_images[label].append(valid_files[i])

    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_directory, exist_ok=True)

    # グループごとにディレクトリを作成し、画像をコピー
    for label, files in grouped_images.items():
        group_directory = os.path.join(output_directory, f"group_{label + 1}")
        os.makedirs(group_directory, exist_ok=True)
        print(f"Creating directory: {group_directory}")
        for file in files:
            try:
                shutil.copy2(file, os.path.join(group_directory, os.path.basename(file)))
                print(f"- Copied {os.path.basename(file)} to {group_directory}")
            except Exception as e:
                print(f"Error copying {file}: {e}")
        print("-" * 20)

if __name__ == "__main__":
    input_directory = "input" # 入力ディレクトリのパス
    output_directory = "output" # 出力ディレクトリのパス
    num_clusters = 3 # グループ数
    group_similar_images_by_color_and_output(input_directory, output_directory, num_clusters)
    print("Image grouping and output complete.")
