import argparse
import os
import warnings
import torch
from PIL import Image
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from tqdm import trange
from transformers import CLIPModel, CLIPProcessor
from hydra_zen import zen, builds, ZenStore

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
# Constants
IMAGE_BASE_FOLDERS = ["/app/data/0/", "/app/data/1/"]

class Processor:
    def __init__(self, model_name: str,
                 qdrant_host: str,
                 qdrant_port: int,
                 collection_name: str = None,
                 compile: bool=True,
                 batch_size:int=32):
        self.model_preprocessor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        if compile:
            self.model = torch.compile(self.model)

        self.db_client = QdrantClient(url=f"http://{qdrant_host}:{qdrant_port}")
        self.collection_name = collection_name
        self.batch_size=batch_size

        self.create_collection()


    def get_image_embedding(self, path: str):
        image = Image.open(path)
        inputs = self.model_preprocessor(images=image, return_tensors="pt")
        outputs = self.model.get_image_features(**inputs)
        return outputs

    def load_batch_to_qdrant(self, paths: list[str],
                             batch: list[list[float]],
                             ids: list[int] = None):
        self.db_client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                payloads=[{"path": path} for path in paths],
                vectors=batch,
                ids=ids
            ),
        )
    def process(self, file_paths: list[str]):
        for batch_start in trange(0, len(file_paths), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(file_paths))
            batch_paths = file_paths[batch_start:batch_end]
            batch_images = [Image.open(path) for path in batch_paths]
            batch_ids = list(range(batch_start, batch_end))

            inputs = self.model_preprocessor(images=batch_images, return_tensors="pt")
            outputs = self.model.get_image_features(**inputs).detach().numpy()

            self.load_batch_to_qdrant(
                paths=batch_paths,
                batch=outputs.tolist(),
                ids=batch_ids
            )

    def create_collection(self):
        try:
            self.db_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=512, distance=models.Distance.COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,
                ),
            )
        except UnexpectedResponse as e:
            print(e)

# Main function
def main(BATCH_SIZE: int = 32,
         model_name: str = "openai/clip-vit-base-patch32",
         qdrant_host: str = "qdrant",
         qdrant_port: int = 6333,
         collection_name: str = "DEFAULT_COLLECTION"
         ):
    file_paths = [
        os.path.join(folder, image)
        for folder in IMAGE_BASE_FOLDERS
        for image in os.listdir(folder)
    ]
    processor = Processor(model_name=model_name,
                          qdrant_host=qdrant_host,
                          qdrant_port=qdrant_port,
                          batch_size=BATCH_SIZE,
                          collection_name=collection_name
                          )
    processor.process(file_paths)


# Entry point
MainConfig = builds(main, populate_full_signature=True)
store = ZenStore()
store(MainConfig, name="load")
if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(main).hydra_main(config_name="load",
                         version_base="1.1",
                         config_path="./config",
                         )
