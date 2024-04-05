# This demo needs to be run from the repo folder.
# python demo/fake_gan/run.py
import random

import gradio as gr
from qdrant_client import QdrantClient
from transformers import CLIPModel, CLIPProcessor

processor = None
model = None
client = None


def fake_gan(input_str):
    processed_input = processor(text=input_str, return_tensors="pt")
    query = model.get_text_features(**processed_input)
    query = query.detach().tolist()[0]
    res = client.search(
        collection_name="AdsDataset",
        query_vector=query,
        with_vectors=True,
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
    btn.click(fake_gan, inputs=textbox, outputs=gallery)


def boot():
    model_name = "openai/clip-vit-base-patch32"

    global processor, model, client

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    client = QdrantClient(url="http://localhost:6333")


if __name__ == "__main__":
    boot()
    demo.launch()
print("Pupa")
import numpy as np