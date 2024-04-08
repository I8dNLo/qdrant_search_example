import logging
import warnings
import os
import gradio as gr
from qdrant_client import QdrantClient
import torch
from transformers import CLIPModel, CLIPProcessor

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

logger = logging.getLogger(__name__)

def_model = os.getenv("MODEL_NAME")
qdrant_port = os.getenv("QDRANT_PORT")
qdrant_host = os.getenv("QDRANT_HOST")
collection_name = os.getenv("COLLECTION_NAME")

frontend_name = os.getenv("FRONTEND_HOST") #"0.0.0.0"
frontend_port = os.getenv("FRONTEND_PORT") #7860

processor = None
model = None
client = None


def search_by_query(input_str: str):
    processed_input = processor(text=input_str, return_tensors="pt")
    query = model.get_text_features(**processed_input)
    query = query.detach().tolist()[0]
    res = client.search(
        collection_name=collection_name,
        query_vector=query,
        with_vectors=False,
        with_payload=True,
    )

    return [elem.payload["path"] for elem in res[:6]]


with gr.Blocks() as demo:
    textbox = gr.Textbox(
        label="Search through the Advertisement Image Dataset",
        info="Text promt",
        lines=1,
        value="Pizza",
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
    btn.click(search_by_query, inputs=textbox, outputs=gallery)


def boot(model_name: str = def_model):
    global processor, model, client

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model = torch.compile(model)
    client = QdrantClient(url=f"http://{qdrant_host}:{qdrant_port}")


if __name__ == "__main__":
    boot()
    print(frontend_name)
    print(frontend_port)
    demo.launch(
        server_name=frontend_name,
        server_port=frontend_port,
        share=True,
    )
