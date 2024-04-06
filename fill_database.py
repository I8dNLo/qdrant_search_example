import os
import argparse

from PIL import Image
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from transformers import CLIPModel, CLIPProcessor
from tqdm import trange
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
# Constants
IMAGE_BASE_FOLDERS = ["./0", "./1/"]
DEFAULT_BATCH_SIZE = 32
DEFAULT_N_SAMPLES = 10000

# Initialize CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Function to get image embeddings
def get_image_embedding(path):
    image = Image.open(path)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.get_image_features(**inputs)
    return outputs


# Function to load batch to Qdrant
def load_batch_to_qdrant(client, paths, batch, collection="AdsDataset", ids=None):
    client.upsert(
        collection_name=collection,
        points=models.Batch(
            payloads=[{"path": path} for path in paths], vectors=batch, ids=ids
        ),
    )


# Main function
def main(BATCH_SIZE, n_samples):
    # Initialize Qdrant client
    client = QdrantClient(url="http://qdrant:6333")

    # Create collection if not exists
    try:
        client.create_collection(
            collection_name="AdsDataset",
            vectors_config=models.VectorParams(
                size=512, distance=models.Distance.COSINE
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=0,
            ),
        )
    except UnexpectedResponse as e:
        print(e)

    # Load images and their paths
    file_paths = [
        os.path.join(folder, image)
        for folder in IMAGE_BASE_FOLDERS
        for image in os.listdir(folder)
    ][:n_samples]

    # Process images in batches
    for batch_start in trange(0, len(file_paths), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(file_paths))
        batch_paths = file_paths[batch_start:batch_end]
        batch_images = [Image.open(path) for path in batch_paths]
        batch_ids = list(range(batch_start, batch_end))

        inputs = processor(images=batch_images, return_tensors="pt")
        outputs = model.get_image_features(**inputs).detach().numpy()

        load_batch_to_qdrant(
            client=client, paths=batch_paths, batch=outputs.tolist(), ids=batch_ids
        )


# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images and load to Qdrant.')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Batch size for processing images')
    parser.add_argument('--n_samples', type=int, default=DEFAULT_N_SAMPLES,
                        help='Number of samples to process')

    args = parser.parse_args()

    main(args.batch_size, args.n_samples)
