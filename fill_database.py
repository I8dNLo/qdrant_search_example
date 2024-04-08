import argparse
import os
import warnings
import torch
from PIL import Image
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from tqdm import trange
from transformers import CLIPModel, CLIPProcessor

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
# Constants
IMAGE_BASE_FOLDERS = ["./0", "./1/"]
DEFAULT_BATCH_SIZE = 32
DEFAULT_N_SAMPLES = 10000
collection_name = os.getenv("COLLECTION_NAME") #"AdsDataset"
qdrant_port = os.getenv("QDRANT_PORT") #6333
qdrant_host = os.getenv("QDRANT_HOST") #qdrant
model_name = os.getenv("MODEL_NAME") #"openai/clip-vit-base-patch32"

# Initialize CLIP model and processor
model = CLIPModel.from_pretrained(model_name).eval()
model = torch.compile(model)
processor = CLIPProcessor.from_pretrained(model_name)


# Function to get image embeddings
def get_image_embedding(path: str):
    image = Image.open(path)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.get_image_features(**inputs)
    return outputs


# Function to load batch to Qdrant
def load_batch_to_qdrant(client: QdrantClient,
                         paths: list[str],
                         batch: list[list[float]],
                         collection : str = collection_name,
                         ids: list[int] = None):
    client.upsert(
        collection_name=collection,
        points=models.Batch(
            payloads=[{"path": path} for path in paths],
            vectors=batch,
            ids=ids
        ),
    )


# Main function
def main(BATCH_SIZE: int, n_samples: int):
    # Initialize Qdrant client
    client = QdrantClient(url=f"http://{qdrant_host}:{qdrant_port}")

    # Create collection if not exists
    try:
        client.create_collection(
            collection_name=collection_name,
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
    parser = argparse.ArgumentParser(description="Process images and load to Qdrant.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for processing images",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Number of samples to process",
    )

    args = parser.parse_args()

    main(args.batch_size, args.n_samples)
