from lavis.models import load_model_and_preprocess
import torch
import argparse
from PIL import Image
import random
import numpy as np

import streamlit as st


def load_demo_image():
    img_url = "images/vegetables.png"

    raw_image = Image.open(img_url).convert("RGB")
    return raw_image


def inference(
        image,
        prompt,
        min_len,
        max_len,
        beam_size,
        len_penalty,
        repetition_penalty,
        top_p,
        decoding_method,
        seed,
    ):
    use_nucleus_sampling = decoding_method == "Nucleus sampling"
    print(image, prompt, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, use_nucleus_sampling)
    image = vis_processors["eval"](image).unsqueeze(0).to(device)

    samples = {
        "image": image,
        "prompt": prompt,
    }

    if seed is not None:
        set_seed(seed)

    output = model.generate(
        samples,
        length_penalty=float(len_penalty),
        repetition_penalty=float(repetition_penalty),
        num_beams=beam_size,
        max_length=max_len,
        min_length=min_len,
        top_p=top_p,
        use_nucleus_sampling=use_nucleus_sampling,
    )

    return output[0]


@st.cache_resource
def load_model_cache(_args):
    args = _args

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    print('Loading model...')

    model, vis_processors, _ = load_model_and_preprocess(
        name=args.model_name,
        model_type=args.model_type,
        is_eval=True,
        device=device,
    )
    
    return model, vis_processors, device

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-name", default="blip2_vicuna_instruct")
    parser.add_argument("--model-type", default="vicuna13b")
    args = parser.parse_args()

    st.markdown(
        "<h1 style='text-align: center;'>InstructBLIP Demo</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """<div style='text-align: center;'>
            <a href='https://arxiv.org/abs/2305.06500'>Report</a> | <a href='https://github.com/salesforce/LAVIS/tree/main/projects/instructblip'>Code</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    image_input = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    min_len = st.sidebar.slider(
        "Min Length",
        min_value=1,
        max_value=50,
        value=1,
        step=1
    )

    max_len = st.sidebar.slider(
        "Max Length",
        min_value=10,
        max_value=500,
        value=250,
        step=5
    )

    sampling = st.sidebar.radio(
        "Text Decoding Method",
        options=["Beam search", "Nucleus sampling"],
        index=0
    )

    if sampling == "Nucleus sampling":
        default_seed = 42
        seed = st.sidebar.text_input("Seed", value=str(default_seed))
        try:
            seed = int(seed)
        except ValueError:
            st.warning("Seed must be an integer, found {}. Using default seed {}.".format(seed, default_seed))
            seed = default_seed
    else:
        seed = None

    top_p = st.sidebar.slider(
        "Top p: increase for more diversity",
        min_value=0.5,
        max_value=1.0,
        value=0.9,
        step=0.1
    )

    beam_size = st.sidebar.slider(
        "Beam Size",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )

    len_penalty = st.sidebar.slider(
        "Length Penalty: increase for longer outputs",
        min_value=-1.0,
        max_value=2.0,
        value=1.0,
        step=0.2,
    )

    repetition_penalty = st.sidebar.slider(
        "Repetition Penalty: increase for less repetition",
        min_value=-1.0,
        max_value=3.0,
        value=1.0,
        step=0.2
    )

    if image_input is not None:
        raw_img = Image.open(image_input).convert("RGB")
    else:
        raw_img = load_demo_image()

    w, h = raw_img.size
    scaling_factor = 720 / w
    resized_image = raw_img.resize((int(w * scaling_factor), int(h * scaling_factor)))

    st.image(resized_image, width=360)

    prompt = st.text_area("Prompt:", value=" Show me steps of making a salad using these items.", height=50)

    cap_button = st.button("Generate")

    output = ""
    if cap_button:
        with st.spinner('Loading model...'):
            model, vis_processors, device = load_model_cache(args)
            print('Loading model done!')
        
        with st.spinner('Generating...'):
            output = inference(
                resized_image, prompt,
                min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, sampling,
                seed=seed
            )

    st.text_area(label="Response",value=output, height=300)
