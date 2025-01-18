import os
import torch
import tensorflow as tf
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import argparse

# Suppress TensorFlow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

device = "cuda" if torch.cuda.is_available() else "cpu"

def main(input_sentence):
    # Define the folder paths
    folder_path = "./small_batch_test"
    image_folder_path = "./data"

    # Initialize an empty list to store sentences and file-to-sentence mapping
    sentences_list = []
    file_mapping = []  # To keep track of which file each sentence belongs to

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Check if the file is a .txt file
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                # Read lines from the file and add them to the list
                lines = file.read().splitlines()
                sentences_list.extend(lines)
                file_mapping.extend([filename] * len(lines))  # Map each sentence to the current file

    # Load a pre-trained multilingual SentenceTransformer model
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').to(device)  # Supports Chinese

    # Compute embeddings for the sentence list and input sentence
    sentence_embeddings = model.encode(sentences_list, convert_to_tensor=True, device=device)
    input_embedding = model.encode(input_sentence, convert_to_tensor=True, device=device)

    # Compute cosine similarity between the input and all sentences in the list
    cosine_scores = util.cos_sim(input_embedding, sentence_embeddings)

    # Find the best matching sentence
    best_match_idx = torch.argmax(cosine_scores)
    best_match_sentence = sentences_list[best_match_idx]
    best_match_file = file_mapping[best_match_idx]  # Get the filename of the best match

    # Output the best match and its filename
    print("Input Sentence: ", input_sentence)
    print("Best Match: ", best_match_sentence)
    print("Found in File: ", best_match_file)

    # Find the corresponding image in the ./data folder
    image_prefix = os.path.splitext(best_match_file)[0]  # Remove .txt from the file name
    for image_name in os.listdir(image_folder_path):
        if image_name.startswith(image_prefix):  # Check for matching prefix
            image_path = os.path.join(image_folder_path, image_name)
            print("Image Found: ", image_name)
            
            # Open and show the image using PIL
            image = Image.open(image_path)
            image.show()
            break
    else:
        print("No matching image found in the folder:", image_folder_path)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Find the best matching sentence and associated image.")
    parser.add_argument("--input", type=str, required=True, help="The input sentence to match.")
    args = parser.parse_args()

    # Call the main function with the provided input sentence
    main(args.input)
