import os

from PIL import Image
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from transformers import CLIPModel, CLIPProcessor
from tqdm import trange
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
# Constants
IMAGE_BASE_FOLDERS = ["./0", "./1/"]
BATCH_SIZE = 64

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
def main():
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
    ][:10]

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
    main()
