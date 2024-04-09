import logging
import warnings
import gradio as gr
from qdrant_client import QdrantClient
from transformers import CLIPModel, CLIPProcessor
from hydra_zen import zen, builds, ZenStore
from itertools import islice

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

logger = logging.getLogger(__name__)

class Processor:
    def __init__(self, model_name: str, qdrant_host: str, qdrant_port: int, res_len: int = 6, collection_name: str = None):
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.db_client = QdrantClient(url=f"http://{qdrant_host}:{qdrant_port}")
        self.res_len = res_len
        self.collection_name = collection_name

    def process(self, query: str):
        processed_input = self.processor(text=query, return_tensors="pt")
        query = self.model.get_text_features(**processed_input)
        query = query.detach().tolist()[0]
        res = self.db_client.search(
            collection_name=self.collection_name,
            query_vector=query,
            with_payload=True,
        )
        return [elem.payload["path"] for elem in islice(res, self.res_len)]


def main(server_name: str = "0.0.0.0",
         server_port: int = 7860,
         model_name: str = "openai/clip-vit-base-patch32",
         qdrant_host: str = "qdrant",
         qdrant_port: int = 6333,
         collection_name: str = "DEFAULT_COLLECTION"):
    processor = Processor(model_name, qdrant_host, qdrant_port, collection_name=collection_name)
    with gr.Blocks() as demo:
        textbox = gr.Textbox(
            label="Search through the Advertisement Image Dataset",
            info="Text prompt",
            lines=1,
            value="Burger",
            render=True,
            interactive=True,
        )
        btn = gr.Button("Search", scale=0)
        gallery = gr.Gallery(
            label="Generated images",
            show_label=False,
            elem_id="gallery",
            columns=[3],
            rows=[1],
            object_fit="contain",
            height="auto",
            # render=False,
            interactive=False,
        )
        btn.click(processor.process, inputs=textbox, outputs=gallery)

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=True,
    )


MainConfig = builds(main, populate_full_signature=True)
store = ZenStore()
store(MainConfig, name="search")
if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(main).hydra_main(config_name="search",
                         version_base="1.1",
                         config_path="./config",
                         )