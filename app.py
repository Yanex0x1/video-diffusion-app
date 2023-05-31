#————————————————————Credits——————————————————
#borrowing heavily from deforum stable diffusion


#Overview
#5. Gradio Interface
#1. Setup
#2. Prompts
#3. Video
#4. Run


#————————————————————1.1. Setup————————————————————————

import subprocess, time, gc, os, sys

def setup_environment():
    start_time = time.time()
    print_subprocess = False
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb: 256"
    #PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    use_xformers_for_colab = True
    try:
        ipy = get_ipython()
    except:
        ipy = 'could not get_ipython'
    if 'google.colab' in str(ipy):
        print("..setting up environment")

        # weird hack
        #import torch
        
        all_process = [
            ['git', 'clone', 'https://github.com/deforum-art/deforum-stable-diffusion'],
            ['pip', 'install', 'omegaconf', 'einops==0.4.1', 'pytorch-lightning==1.7.7', 'torchmetrics', 'transformers', 'safetensors', 'kornia'],
            ['pip', 'install', 'accelerate', 'ftfy', 'jsonmerge', 'matplotlib', 'resize-right', 'timm', 'torchdiffeq','scikit-learn','torchsde','open-clip-torch','numpngw'],
        ]
        for process in all_process:
            running = subprocess.run(process,stdout=subprocess.PIPE).stdout.decode('utf-8')
            if print_subprocess:
                print(running)
        with open('deforum-stable-diffusion/src/k_diffusion/__init__.py', 'w') as f:
            f.write('')
        sys.path.extend([
            'deforum-stable-diffusion/',
            'deforum-stable-diffusion/src',
        ])
        if use_xformers_for_colab:

            print("..installing triton and xformers")

            all_process = [['pip', 'install', 'triton==2.0.0.dev20221202', 'xformers==0.0.16']]
            for process in all_process:
                running = subprocess.run(process,stdout=subprocess.PIPE).stdout.decode('utf-8')
                if print_subprocess:
                    print(running)
    else:
        sys.path.extend([
            'src'
        ])
    end_time = time.time()
    print(f"..environment set up in {end_time-start_time:.0f} seconds")
    return

setup_environment()

#————————————————————1.2. Imports————————————————————————

import os
import torch
import random
import clip
import gradio as gr
import re
import random

from base64 import b64encode

#from k_diffusion.external import CompVisDenoiser, CompVisVDenoiser
import py3d_tools as p3dT

from IPython import display
from types import SimpleNamespace
from helpers.save_images import get_output_folder
from helpers.settings import load_args
from helpers.render import render_animation, render_input_video, render_image_batch, render_interpolation
from helpers.model_load import make_linear_decode, load_model, get_model_output_paths
from helpers.aesthetics import load_aesthetics_model
from diffusers import DiffusionPipeline
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from share_btn import community_icon_html, loading_icon_html, share_js
#from plms import PLMSSampler

#from AdaBins-main import infer, model_io, utils
#from AdaBins-main.models import unet_adaptive_bins.py

import gradio as gr
#from datasets import load_dataset
from PIL import Image  

import requests

#————————————————————1.3. Token Setup———————————————

MY_SECRET_TOKEN=os.environ.get('HF_TOKEN_SD')
device = "cuda" if torch.cuda.is_available() else "cpu" #is this needed?

#———————————————————— infer from SD ———————————————

"""
word_list_dataset = load_dataset("stabilityai/word-list", data_files="list.txt", use_auth_token=True)
word_list = word_list_dataset["train"]['text']

is_gpu_busy = False
def infer(prompt, negative, scale):
    global is_gpu_busy
    for filter in word_list:
        if re.search(rf"\b{filter}\b", prompt):
            raise gr.Error("Unsafe content found. Please try again with different prompts.")
        
    images = []
    url = os.getenv('JAX_BACKEND_URL')
    payload = {'prompt': prompt, 'negative_prompt': negative, 'guidance_scale': scale}
    images_request = requests.post(url, json = payload)
    for image in images_request.json()["images"]:
        image_b64 = (f"data:image/jpeg;base64,{image}")
        images.append(image_b64)
    
    return images
    """


#————————————————————5.1 Gradio Interface ———————————————

#CSS defines the style of the interface
css = """
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: black;
            background: black;
        }
        input[type='range'] {
            accent-color: black;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 730px;
            margin: auto;
            padding-top: 1.5rem;
        }
     
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: #2596be; #changed this from: white;
        }
        .dark .footer {
            border-color: #303030;
            
        }
        .dark .footer>p {
            background: #2596be; #changed this from 0b0f19;
        }
        .prompt h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
        .animate-spin {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        #share-btn-container {
            display: flex; margin-top: 1.5rem !important; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
        }
        #share-btn {
            all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;
        }
        #share-btn * {
            all: unset;
        }
"""

#creates the interface object with the style outlined above
block = gr.Blocks(css=css)

#HTML defines the layout of the interface
with block:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 650px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <svg
                  width="0.65em"
                  height="0.65em"
                  viewBox="0 0 115 115"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <rect width="23" height="23" fill="white"></rect>
                  <rect y="69" width="23" height="23" fill="white"></rect>
                  <rect x="23" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="46" width="23" height="23" fill="white"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" width="23" height="23" fill="black"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="92" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="115" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="46" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="46" y="46" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="115" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="23" y="46" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="23" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="black"></rect>
                </svg>
                <h1 style="font-weight: 900; margin-bottom: 7px;">
                  Hallucinate
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
                Instantly produce cinematics for your audio. &#10;&#13;
                Create unique Spotify Canvases for all your tracks. &#10;&#13;
              </p> 
            </div>
        """
    )
    
    #within the group
    with gr.Group():

        #first create a box
        with gr.Box():

            #in the box create a row
            with gr.Row().style(mobile_collapse=False, equal_height=True):

                #in the row add a video input (left) #UPDATE THIS
                image_input = gr.Image(
                    label="Initialize with image",
                    show_label=False,
                    source="upload", #"microphone"
                    type="filepath"
                )

                #in the row add a button to run the model (right) #UPDATE THIS
                btn = gr.Button("Hallucinate")

        #add an output field to the box      #UPDATE THIS
        video_output = gr.Video(show_label=False, elem_id="result-textarea")
        
        #add share functions
        with gr.Group(elem_id="share-btn-container"):
            community_icon = gr.HTML(community_icon_html, visible=False)
            loading_icon = gr.HTML(loading_icon_html, visible=False)
            share_button = gr.Button("Share to community", elem_id="share-btn", visible=False)
                                    
        
        #add button functions
            
    #the input button is here
            
        #btn.click(render_input_video, inputs=[video_input], outputs=[video_output, community_icon, loading_icon, share_button])
        #share_button.click(None, [], [], _js=share_js)

        #create footer
        gr.HTML(
            """
                <div class="footer">
                    <p>Experience by <a href="https://www.hallucinate.it/" style="text-decoration: underline;" target="_blank">Hallucinate</a> - Gradio Demo by 🤗 Hugging Face
                    </p>
                </div>
           """
        )
        with gr.Accordion(label="License", open=False):
            gr.HTML(
                """<div class="acknowledgments">
                    <p><h4>LICENSE</h4>
The model is licensed with a <a href="https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL" style="text-decoration: underline;" target="_blank">CreativeML OpenRAIL++</a> license. The authors claim no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in this license. The license forbids you from sharing any content that violates any laws, produce any harm to a person, disseminate any personal information that would be meant for harm, spread misinformation and target vulnerable groups. For the full list of restrictions please <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" target="_blank" style="text-decoration: underline;" target="_blank">read the license</a></p>
                    <p><h4>Biases and content acknowledgment</h4>
Despite how impressive being able to turn text into image is, beware to the fact that this model may output content that reinforces or exacerbates societal biases, as well as realistic faces, pornography and violence. The model was trained on the <a href="https://laion.ai/blog/laion-5b/" style="text-decoration: underline;" target="_blank">LAION-5B dataset</a>, which scraped non-curated image-text-pairs from the internet (the exception being the removal of illegal content) and is meant for research purposes. You can read more in the <a href="https://huggingface.co/CompVis/stable-diffusion-v1-4" style="text-decoration: underline;" target="_blank">model card</a></p>
               </div>
                """
            )

#launch
#block.launch()#share=True)






#———————————————————— 1.4. Path Setup (input, output, drive)————————————————————————

#this creates FileNotFoundError: [Errno 2] No such file or directory: 
#/home/user/app/configs/v1-inference.yaml - is this a github link?
#even though the yaml file is uploaded to files. yaml file should be available through DSD import?

def Root():
    models_path = "models" #@param {type:"string"}
    configs_path = "configs" #@param {type:"string"}
    output_path = "outputs" #@param {type:"string"}
    mount_google_drive = True #@param {type:"boolean"}
    models_path_gdrive = "/content/drive/MyDrive/AI/models" #@param {type:"string"}
    output_path_gdrive = "/content/drive/MyDrive/AI/StableDiffusion" #@param {type:"string"}

    #@markdown **Model Setup**
    map_location = "cuda" #@param ["cpu", "cuda"]
    model_config = "v1-inference.yaml" #@param ["custom","v2-inference.yaml","v2-inference-v.yaml","v1-inference.yaml"]
    model_checkpoint =  "Protogen_V2.2.ckpt" #@param ["custom","v2-1_768-ema-pruned.ckpt","v2-1_512-ema-pruned.ckpt","768-v-ema.ckpt","512-base-ema.ckpt","Protogen_V2.2.ckpt","v1-5-pruned.ckpt","v1-5-pruned-emaonly.ckpt","sd-v1-4-full-ema.ckpt","sd-v1-4.ckpt","sd-v1-3-full-ema.ckpt","sd-v1-3.ckpt","sd-v1-2-full-ema.ckpt","sd-v1-2.ckpt","sd-v1-1-full-ema.ckpt","sd-v1-1.ckpt", "robo-diffusion-v1.ckpt","wd-v1-3-float16.ckpt"]
    custom_config_path = "" #"https://github.com/realhallucinate/deforum-stable-diffusion-gradioUI/blob/main/configs/v1-inference.yaml"# replaced just an empty string: "" with a diret link #@param {type:"string"}
    custom_checkpoint_path = "" #@param {type:"string"}
    return locals()

root = Root()
root = SimpleNamespace(**root)

root.models_path, root.output_path = get_model_output_paths(root)
root.model, root.device = load_model(root, load_on_run_all=True, check_sha256=True, map_location=root.map_location)
#root.model, root.device = load_model(root, load_on_run_all=True, check_sha256=True, map_location=torch.device('cpu'))


#——————————————————— 2.1. Prompt Base ———————————————————————— #----------------CONSIDER THIS FOR INPUT/LOCK---------------

#need to update the prompt base

medium = {"pixar image | matte painting | 3D render | oil painting | photograph | sculpture |digital illustration | watercolor | colored pencil sketch | photo shoot | acrylic painting"}
description = {"silly | sexy | golden | exotic | beautiful | elegant | creepy | hilarious | evil | Angelic | sublime | ridiculous"}
subject = {"rococo cauliflower headdress | cauliflower cornucopia | macro romanesco | cauliflower Cthulhu | cauliflower nuclear explosion | Cauliflower mushroom cloud | Hubble cauliflower nebula | cauliflower infestation | steampunk cauliflower | magic rubber cauliflower | psychedelic cauliflower | cauliflower couture"}
details = {"flowers | ornaments | pearls | raindrops | glasses"}
artist = {"[3$$ James Jean | Lucian Freud | tomm moore  | wes anderson  | ernst haeckl | tim burton  | jean pierre jeunet | jean giraud moebius | takashi murakami | ross draws |  artgerm |  alvin ailey |  Zdzisław Beksiński |  Arthur Rackham |  Dariusz Zawadzki |  thomas kincade  |  greg rutkowski  |  rembrandt |  HR Giger |  jama jurabaev | wenwei zhu | frank franzzeta | mcbess | sakimi chan | brosmind | steve simpson | jim henson | Nicoletta Ceccoli | Margaret Keane | Mark Ryden | Severin Krøyer | Marie Krøyer | Karl Madsen| Laurits Tuxen| Carl Locher| Viggo Johansen| Thorvald Niss | Holger Drachmann | Raluca bararu | walton ford | josh Keyes | Marco Mazzoni | Susan Helen Strok | R Crumb | Beatrix potter | shaw brothers | victor moscoso | Thomas Kinkade | Vincent Van Gogh | Leonid Afremov | Claude Monet | Edward Hopper | Norman Rockwell | William-Adolphe Bouguereau | Albert Bierstadt | John Singer Sargent | Pierre-Auguste Renoir | Frida Kahlo | John William Waterhouse | Winslow Homer | Walt Disney | Thomas Moran | Phil Koch | Paul Cézanne | Camille Pissarro | Erin Hanson | Thomas Cole | Raphael | Steve Henderson | Pablo Picasso | Caspar David Friedrich | Ansel Adams | Diego Rivera | Steve McCurry | Bob Ross | John Atkinson Grimshaw | Rob Gonsalves | Paul Gauguin | James Tissot | Edouard Manet | Alphonse Mucha | Alfred Sisley | Fabian Perez | Gustave Courbet | Zaha Hadid | Jean-Léon Gérôme | Carl Larsson | Mary Cassatt | Sandro Botticelli | Daniel Ridgway Knight | Joaquín Sorolla | Andy Warhol | Kehinde Wiley | Alfred Eisenstaedt | Gustav Klimt | Dante Gabriel Rossetti | Tom Thomson | Edgar Degas | Utagawa Hiroshige | Camille Corot | Edward Steichen | David Hockney | Ivan Aivazovsky | Josephine Wall | Peter Paul Rubens | Henri Rousseau | Edward Burne-Jones | Pixar | Alexander McQueen | Anders Zorn | Jean Auguste Dominique Ingres | Franz Xaver Winterhalter | Katsushika Hokusai | John Constable | Canaletto | Shepard Fairey | Gordon Parks | George Inness | Anthony van Dyck | Vivian Maier | Catrin Welz-Stein | Lawren Harris | Salvador Dali | David Bowie | Agnes Cecile | Titian | Martin Johnson Heade | Scott Naismith | William Morris | Berthe Morisot | Vladimir Kush | William Holman Hunt | Edvard Munch | Joseph Mallord William Turner | Gustave Doré | Thomas Eakins | Ilya Repin | Amedeo Modigliani | Johannes Vermeer | Eyvind EarleIvan Shishkin | Rembrandt Van Rijn | Gil Elvgren | Nicholas Roerich | Henri Matisse | Thomas Gainsborough | Artgerm | Studio Ghibli | Grant Wood | Jeremy Mann | Mark Keathley | Maxfield Parrish | Andrew Wyeth | RHADS | David Lynch | Frederic Remington | Jan Van Eyck | Mikko Lagerstedt | Banksy | Michael Cheval | Anna Razumovskaya | Jean-François Millet | Thomas W Schaller | Charlie Bowater | El Greco | Paolo Roversi | Carne Griffiths | Man Ray | August Sander | Andrew Macara | Evelyn De Morgan | William Blake | Sally Mann | Oleg Oprisco | Yuumei | Helmut Newton | Henry Ossawa Tanner | Asher Brown Durand | teamLab | August Macke | Armand Guillaumin | Terry Redlin | Antoine Blanchard | Anna Ancher | Ohara Koson | Walter Langley | Yayoi Kusama | Stan Lee | Chuck Close | Albert Edelfelt | Mark Seliger | Eugene Delacroix | John Lavery | Theo van Rysselberghe | Marc Chagall | Rolf Armstrong | Brent Heighton | A.J.Casson | Egon Schiele | Maximilien Luce | Georges Seurat | George Frederic Watts | Arthur Hughes | Anton Mauve | Lucian Freud | Jessie Willcox Smith | Leonardo Da Vinci | Edward John Poynter | Brooke Shaden | J.M.W. Turner | Wassily Kandinsky | Wes Anderson | Jean-Honoré Fragonard | Amanda Clark | Tom Roberts | Antonello da Messina | Makoto Shinkai | Hayao Miyazaki | Slim Aarons | Alfred Stevens | Albert Lynch | Andre Kohn | Daniel Garber | Jacek Yerka | Beatrix Potter | Rene Magritte | Georgia O'Keeffe | Isaac Levitan | Frank Lloyd Wright | Gustave Moreau | Ford Madox Brown | Ai Weiwei | Tim Burton | Alfred Cheney Johnston | Duy Huynh | Michael Parkes | Tintoretto | Archibald Thorburn | Audrey Kawasaki | George Lucas | Arthur Streeton | Albrecht Durer | Andrea Kowch | Dorina Costras | Alex Ross | Hasui Kawase | Lucas Cranach the Elder | Briton Rivière | Antonio Mora | Mandy Disher | Henri-Edmond Cross | Auguste Toulmouche | Hubert Robert | Syd Mead | Carl Spitzweg | Alyssa Monks | Edward Lear | Ralph McQuarrie | Sailor Moon | Simon Stalenhag | Edward Robert Hughes | Jules Bastien-Lepage | Richard S. Johnson | Rockwell Kent | Sparth | Arnold Böcklin | Lovis Corinth | Arnold Bocklin | Robert Hagan | Gregory Crewdson | Thomas Benjamin Kennington | Abbott Handerson Thayer | Gilbert Stuart | Louis Comfort Tiffany | Raphael Lacoste | Jean Marc Nattier | Janek Sedlar | Sherree Valentine Daines | Alexander Jansson | James Turrell | Alex Grey | Henri De Toulouse Lautrec | Anton Pieck | Ramon Casas | Andrew Atroshenko | Andy Kehoe | Andreas Achenbach | H.P. Lovecraft | Eric Zener | Kunisada | Jimmy Lawlor | Quentin Tarantino | Marianne North | Vivienne Westwood | Tom Bagshaw | Jeremy Lipking | John Martin | Cindy Sherman | Scott Listfield | Alexandre Cabanel | Arthur Rackham | Arthur Hacker | Henri Fantin Latour | Mark Ryden | Peter Holme IIIT | ed Nasmith | Bill Gekas | Paul Strand | Anne Stokes | David Teniers the Younger | Alan Lee | Ed Freeman | Andrey Remnev | Alasdair McLellan | Botero | Vittorio Matteo Corcos | Ed Mell | Worthington Whittredge | Jakub Różalski | Alex Gross | Edward Weston | Ilya Kuvshinov | Francisco De Goya | Balthus | J.C. Leyendecker | Nathan Wirth]"}
realism = {"[4$$ highly detailed | photorealistic | realistic | hypermaximalist | hyperrealism, intricate | HD | 8k | 4k | octane render | subsurface scattering raytracing | depth of field | bokeh | redshift render | abstract illusionism | Atmospheric | Dramatic lighting | Anthropomorphic | 8k | Very detailed | Cinematic lighting | Unreal engine | Octane render | Photorealistic | Hyperrealistic | Sharp focus | Rim lighting | Soft lighting | Volumetric | Surreal | Realistic CGI | Fantastic backlight | HDR | Studio light | Internal glow | Iridescent | Cyberpunk | Steampunk | Intricate  | filigree | Bionic futurism | Ray tracing | Symmetrical | Atompunk | Multiverse | Concept art | Time loop | Maximum texture | Futurism | Dynamic | retrowave | afrofuturism | intricate and highly detailed |  intricate and highly detailed |  intricate and highly detailed |  intricate and highly detailed |  intricate and highly detailed | photorealistic |  photorealistic |  photorealistic |  photorealistic]"}
repository = {"Artstation"}
#setting = {"corporate office setting | abandoned warehouse | schoolhouse | victorian train station | victorian room | Lake | Field of wild flowers | submarine | tennis court | mcdonalds | swamp | assembly line | surface of the moon | museum | outer space |storefront display"}
#time = {"morning | noon | night | evening | dawn"}

#'animation_mode: None' (default) batches on this list of 'prompts'
prompts = [
    f"A beautiful {medium} of a {description}{subject} with {details}, in the style of {artist}. {realism} design, trending on {repository}"
]

#——————————————————— 2.2. Prompt Template Builder ————————————————————————

#a function to select a set of prompts

# Define the `pick_variant` function that takes in a template string
def pick_variant(template):
  # If the template is None, return None
  if template is None:
    return None

  # Set `out` to be the template string
  out = template
  
  # Use a regular expression to find all instances of `{...}` in the template
  # The regular expression `r'\{[^{}]*?}'` searches for all sequences of characters
  # surrounded by curly braces that do not contain other curly braces.
  variants = re.findall(r'\{[^{}]*?}', out)

  # Iterate over each variant found in the template
  for v in variants:
    # Split the variant string by the vertical bar (|)
    opts = v.strip("{}").split('|')
    # Replace the variant in the `out` string with a random option
    out = out.replace(v, random.choice(opts))
  
  # Use a regular expression to find all instances of `[...]` in the template
  # The regular expression `r'\[[^\[\]]*?]'` searches for all sequences of characters
  # surrounded by square brackets that do not contain other square brackets.
  combinations = re.findall(r'\[[^\[\]]*?]', out)

  # Iterate over each combination found in the template
  for c in combinations:
    # Remove the square brackets from the combination
    sc = c.strip("[]")
    # Split the combination string by '$$'
    parts = sc.split('$$')
    # Initialize the number of options to pick to `None`
    n_pick = None

    # If there are more than 2 parts, raise an error
    if len(parts) > 2:
      raise ValueError(" we do not support more than 1 $$ in a combination")
    # If there are 2 parts, set the number of options to pick to the first part
    if len(parts) == 2:
      sc = parts[1]
      n_pick = int(parts[0]) 
    # Split the combination string by the vertical bar (|)
    opts = sc.split('|')
    # If the number of options to pick is not set, set it to a random integer between 1 and the number of options
    if not n_pick:
      n_pick = random.randint(1,len(opts))

    # Sample `n_pick` options from the options list
    sample = random.sample(opts, n_pick)
    # Replace the combination in the `out` string with a comma-separated string of the picked options
    out = out.replace(c, ', '.join(sample))

  # If there were any variants or combinations found in the template, call `pick_variant` again with the new `out` string
  if len(variants+combinations) > 0:
    return pick_variant(out)
  # If there were no variants or combinations found, return the final `out` string
  return out

#'animation_mode: None' (default) batches on this list of 'prompts'
for prompt in prompts:
  text_prompt = pick_variant(prompt)

#print('Text prompt selected:', "\n")
#print(text_prompt)

#———————————————————— 2.3. Prompt Selector: Video ————————————————————————

#create a string of frame intervals and prompts
def prompt_intervals(prompts, framecount, stepsize):

  timesteps = []
  for frame in range(0, framecount):
    timesteps.append(frame * stepsize)

  animation_prompt = ""
  for timestep in timesteps:
    for prompt in prompts:
      p = pick_variant(prompt)
      animation_prompt += (str(timestep) + ": " + p + ", ")
  
  animation_prompts = str(animation_prompt)
  return animation_prompts

#@markdown Here you can select framecount and stepsize and create a selection of animation prompts

#MAKE INTERACTABLE IN DEMO? OR HAVE FIXED 8 FPS?
framecount = 8 #fps #----------------CONSIDER THIS FOR INPUT/LOCK---------------
stepsize = 25 #time interval between prompts #----------------CONSIDER THIS FOR INPUT/LOCK---------------

#'animation_mode: 2D' works with this list of 'animation_prompts'
animation_prompts = prompt_intervals(prompts, framecount, stepsize)

#print('Animation prompts selected:',"\n")
#print(animation_prompts)


#———————————————————— 3.1. Video Settings————————————————————————
    #HARDCODE THE DIFFERENT CAMERA SETTINGS HERE LATER?

#This function only sets and outputs the arguments for the ANIMATION process
def DeforumAnimArgs():

    #ANIMATION_MODE IS A KEY ARG!
    
    #@markdown ####**Animation:**
    animation_mode = '3D' #@param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'} #THIS IS A KEY ARG!
    max_frames = 1000 #@param {type:"number"}
    border = 'replicate' #@param ['wrap', 'replicate'] {type:'string'}

    #@markdown ####**Motion Parameters:**
    angle = "0:(0)"#@param {type:"string"} #----------------CONSIDER THIS FOR INPUT/LOCK---------------
    zoom = "0:(1.04)"#@param {type:"string"} #----------------CONSIDER THIS FOR INPUT/LOCK---------------
    translation_x = "0:(10*sin(2*3.14*t/10))"#@param {type:"string"} #----------------CONSIDER THIS FOR INPUT/LOCK---------------
    translation_y = "0:(0)"#@param {type:"string"} #----------------CONSIDER THIS FOR INPUT/LOCK---------------
    translation_z = "0:(10)"#@param {type:"string"}#----------------CONSIDER THIS FOR INPUT/LOCK---------------
    rotation_3d_x = "0:(0)"#@param {type:"string"} #----------------CONSIDER THIS FOR INPUT/LOCK---------------
    rotation_3d_y = "0:(0)"#@param {type:"string"} #----------------CONSIDER THIS FOR INPUT/LOCK---------------
    rotation_3d_z = "0:(0)"#@param {type:"string"} #----------------CONSIDER THIS FOR INPUT/LOCK---------------
    flip_2d_perspective = False #@param {type:"boolean"}
    perspective_flip_theta = "0:(0)"#@param {type:"string"}
    perspective_flip_phi = "0:(t%15)"#@param {type:"string"}
    perspective_flip_gamma = "0:(0)"#@param {type:"string"}
    perspective_flip_fv = "0:(53)"#@param {type:"string"}
    noise_schedule = "0: (0.02)"#@param {type:"string"}
    strength_schedule = "0: (0.65)"#@param {type:"string"}
    contrast_schedule = "0: (1.0)"#@param {type:"string"}
    hybrid_video_comp_alpha_schedule = "0:(1)" #@param {type:"string"}
    hybrid_video_comp_mask_blend_alpha_schedule = "0:(0.5)" #@param {type:"string"}
    hybrid_video_comp_mask_contrast_schedule = "0:(1)" #@param {type:"string"}
    hybrid_video_comp_mask_auto_contrast_cutoff_high_schedule =  "0:(100)" #@param {type:"string"}
    hybrid_video_comp_mask_auto_contrast_cutoff_low_schedule =  "0:(0)" #@param {type:"string"}

    #@markdown ####**Unsharp mask (anti-blur) Parameters:**
    kernel_schedule = "0: (5)"#@param {type:"string"}
    sigma_schedule = "0: (1.0)"#@param {type:"string"}
    amount_schedule = "0: (0.2)"#@param {type:"string"}
    threshold_schedule = "0: (0.0)"#@param {type:"string"}

    #@markdown ####**Coherence:**
    color_coherence = 'Match Frame 0 LAB' #@param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB', 'Video Input'] {type:'string'}
    color_coherence_video_every_N_frames = 1 #@param {type:"integer"}
    diffusion_cadence = '1' #@param ['1','2','3','4','5','6','7','8'] {type:'string'}

    #@markdown ####**3D Depth Warping:**
    use_depth_warping = True #@param {type:"boolean"}
    midas_weight = 0.3#@param {type:"number"}
    near_plane = 200
    far_plane = 10000
    fov = 40#@param {type:"number"}
    padding_mode = 'border'#@param ['border', 'reflection', 'zeros'] {type:'string'}
    sampling_mode = 'bicubic'#@param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
    save_depth_maps = False #@param {type:"boolean"}

#video input here
    
    #@markdown ####**Video Input:**
    video_init_path = image_input #'/content/video_in.mp4'#@param {type:"string"} #----------------CONSIDER THIS FOR INPUT/LOCK---------------
    extract_nth_frame = 1#@param {type:"number"}
    overwrite_extracted_frames = True #@param {type:"boolean"}
    use_mask_video = False #@param {type:"boolean"}
    video_mask_path ='/content/video_in.mp4'#@param {type:"string"}

    #@markdown ####**Hybrid Video for 2D/3D Animation Mode:**
    hybrid_video_generate_inputframes = False #@param {type:"boolean"}
    hybrid_video_use_first_frame_as_init_image = True #@param {type:"boolean"} #----------------CONSIDER THIS FOR INPUT/LOCK---------------
    hybrid_video_motion = "None" #@param ['None','Optical Flow','Perspective','Affine']
    hybrid_video_flow_method = "Farneback" #@param ['Farneback','DenseRLOF','SF']
    hybrid_video_composite = False #@param {type:"boolean"}
    hybrid_video_comp_mask_type = "None" #@param ['None', 'Depth', 'Video Depth', 'Blend', 'Difference']
    hybrid_video_comp_mask_inverse = False #@param {type:"boolean"}
    hybrid_video_comp_mask_equalize = "None" #@param  ['None','Before','After','Both']
    hybrid_video_comp_mask_auto_contrast = False #@param {type:"boolean"}
    hybrid_video_comp_save_extra_frames = False #@param {type:"boolean"}
    hybrid_video_use_video_as_mse_image = False #@param {type:"boolean"}

    #@markdown ####**Interpolation:**
    interpolate_key_frames = False #@param {type:"boolean"}
    interpolate_x_frames = 4 #@param {type:"number"}
    
    #@markdown ####**Resume Animation:**
    resume_from_timestring = False #@param {type:"boolean"}
    resume_timestring = "20220829210106" #@param {type:"string"}

    return locals()    


#———————————————————— 4.1. Run (create and return images)————————————————————————
    
#@markdown **Load Settings**
override_settings_with_file = False #@param {type:"boolean"}
settings_file = "custom" #@param ["custom", "512x512_aesthetic_0.json","512x512_aesthetic_1.json","512x512_colormatch_0.json","512x512_colormatch_1.json","512x512_colormatch_2.json","512x512_colormatch_3.json"]
custom_settings_file = "/content/drive/MyDrive/Settings.txt"#@param {type:"string"}

#This function only sets and outputs the arguments for the INFERENCE process
def DeforumArgs():
    #@markdown **Image Settings**
    W = 540 #@param
    H = 540 #@param
    W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64
    bit_depth_output = 8 #@param [8, 16, 32] {type:"raw"}

    #@markdown **Sampling Settings**
    seed = 1 #@param 
    sampler = 'euler_ancestral' #@param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim", "dpm_fast", "dpm_adaptive", "dpmpp_2s_a", "dpmpp_2m"]
    steps = 25 #@param #----------------CONSIDER THIS FOR INPUT/LOCK---------------
    scale = 7 #@param
    ddim_eta = 0.0 #@param
    dynamic_threshold = None
    static_threshold = None   

    #@markdown **Save & Display Settings**
    save_samples = False #@param {type:"boolean"} #----------------CONSIDER THIS FOR INPUT/LOCK---------------
    save_settings = False #@param {type:"boolean"}
    display_samples = False #@param {type:"boolean"} #----------------CONSIDER THIS FOR INPUT/LOCK---------------
    save_sample_per_step = False #@param {type:"boolean"}
    show_sample_per_step = False #@param {type:"boolean"}

    #@markdown **Prompt Settings**
    prompt_weighting = True #@param {type:"boolean"}
    normalize_prompt_weights = True #@param {type:"boolean"}
    log_weighted_subprompts = False #@param {type:"boolean"}

    #@markdown **Batch Settings**
    n_batch = 12 #@param #----------------CONSIDER THIS FOR INPUT/LOCK---------------
   
    batch_name = "HuggingTest1" #@param {type:"string"} #----------------CONSIDER THIS FOR INPUT/LOCK---------------
    filename_format = "{timestring}_{index}_{prompt}.png" #@param ["{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"]
    seed_behavior = "iter" #@param ["iter","fixed","random","ladder","alternate"] #----------------CONSIDER THIS FOR INPUT/LOCK---------------
    seed_iter_N = 1 #@param {type:'integer'} #----------------CONSIDER THIS FOR INPUT/LOCK---------------
    make_grid = False #@param {type:"boolean"}
    grid_rows = 2 #@param 
    outdir = get_output_folder(root.output_path, batch_name) #----------------CONSIDER THIS FOR INPUT/LOCK---------------

    #@markdown **Init Settings**
    use_init = False #@param {type:"boolean"} #----------------CONSIDER THIS FOR INPUT/LOCK---------------
    strength = 0.75 #@param {type:"number"} #----------------CONSIDER THIS FOR INPUT/LOCK---------------
    strength_0_no_init = True # Set the strength to 0 automatically when no init image is used
    init_image = "/content/drive/MyDrive/AI/init_images/Hallucinate.png" #@param {type:"string"} #----------------CONSIDER THIS FOR INPUT/LOCK---------------
    # Whiter areas of the mask are areas that change more
    use_mask = False #@param {type:"boolean"} #----------------CONSIDER THIS FOR INPUT/LOCK---------------
    use_alpha_as_mask = False # use the alpha channel of the init image as the mask
    mask_file = "https://www.filterforge.com/wiki/images/archive/b/b7/20080927223728%21Polygonal_gradient_thumb.jpg" #@param {type:"string"}
    invert_mask = False #@param {type:"boolean"}
    # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
    mask_brightness_adjust = 1.0  #@param {type:"number"}
    mask_contrast_adjust = 1.0  #@param {type:"number"}
    # Overlay the masked image at the end of the generation so it does not get degraded by encoding and decoding
    overlay_mask = True  # {type:"boolean"}
    # Blur edges of final overlay mask, if used. Minimum = 0 (no blur)
    mask_overlay_blur = 5 # {type:"number"}

    #@markdown **Exposure/Contrast Conditional Settings**
    mean_scale = 0 #@param {type:"number"}
    var_scale = 0 #@param {type:"number"}
    exposure_scale = 0 #@param {type:"number"}
    exposure_target = 0.5 #@param {type:"number"}

    #@markdown **Color Match Conditional Settings**
    colormatch_scale = 0 #@param {type:"number"}
    colormatch_image = "https://www.saasdesign.io/wp-content/uploads/2021/02/palette-3-min-980x588.png" #@param {type:"string"}
    colormatch_n_colors = 4 #@param {type:"number"}
    ignore_sat_weight = 0 #@param {type:"number"}

    #@markdown **CLIP\Aesthetics Conditional Settings**
    clip_name = 'ViT-L/14' #@param ['ViT-L/14', 'ViT-L/14@336px', 'ViT-B/16', 'ViT-B/32']
    clip_scale = 0 #@param {type:"number"}
    aesthetics_scale = 0 #@param {type:"number"}
    cutn = 1 #@param {type:"number"}
    cut_pow = 0.0001 #@param {type:"number"}

    #@markdown **Other Conditional Settings**
    init_mse_scale = 0 #@param {type:"number"}
    init_mse_image = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg" #@param {type:"string"}

    blue_scale = 0 #@param {type:"number"}
    
    #@markdown **Conditional Gradient Settings**
    gradient_wrt = 'x0_pred' #@param ["x", "x0_pred"]
    gradient_add_to = 'both' #@param ["cond", "uncond", "both"]
    decode_method = 'linear' #@param ["autoencoder","linear"]
    grad_threshold_type = 'dynamic' #@param ["dynamic", "static", "mean", "schedule"]
    clamp_grad_threshold = 0.2 #@param {type:"number"}
    clamp_start = 0.2 #@param
    clamp_stop = 0.01 #@param
    grad_inject_timing = list(range(1,10)) #@param

    #@markdown **Speed vs VRAM Settings**
    cond_uncond_sync = True #@param {type:"boolean"}

    n_samples = 1 # doesnt do anything
    precision = 'autocast' 
    C = 4
    f = 8

    prompt = ""
    timestring = ""
    init_latent = None
    init_sample = None
    init_sample_raw = None
    mask_sample = None
    init_c = None
    seed_internal = 0

    return locals()

#This segment prepares arguments and adjusts settings

# Define default arguments for the program
args_dict = DeforumArgs()
anim_args_dict = DeforumAnimArgs()

# Override default arguments with values from settings file, if specified
if override_settings_with_file:
    load_args(args_dict, anim_args_dict, settings_file, custom_settings_file, verbose=False)

# Create SimpleNamespace objects for arguments and animation arguments
args = SimpleNamespace(**args_dict)
anim_args = SimpleNamespace(**anim_args_dict)

# Set timestring to current time in YYYYMMDDHHMMSS format
args.timestring = time.strftime('%Y%m%d%H%M%S')
# Ensure strength is within valid range of 0.0 to 1.0
args.strength = max(0.0, min(1.0, args.strength))

# Load clip model if using clip guidance or aesthetics model if aesthetics_scale is > 0
if (args.clip_scale > 0) or (args.aesthetics_scale > 0):
    # Load clip model and set to evaluation mode without requiring gradient
    root.clip_model = clip.load(args.clip_name, jit=False)[0].eval().requires_grad_(False).to(root.device)
    if (args.aesthetics_scale > 0):
        # Load aesthetics model if aesthetics_scale is > 0
        root.aesthetics_model = load_aesthetics_model(args, root)

# Set seed to a random number if not specified
if args.seed == -1:
    args.seed = random.randint(0, 2**32 - 1)

# If not using init image, set init_image to None
if not args.use_init:
    args.init_image = None

# If using plms sampler with init image or animation mode isn't None, switch to klms sampler
if args.sampler == 'plms' and (args.use_init or anim_args.animation_mode != 'None'):
    print(f"Init images aren't supported with PLMS yet, switching to KLMS")
    args.sampler = 'klms'

# If not using ddim sampler, set ddim_eta to 0
if args.sampler != 'ddim':
    args.ddim_eta = 0

# Set max_frames to 1 if animation mode is None, or use_init to True if animation mode is Video Input
if anim_args.animation_mode == 'None':
    anim_args.max_frames = 1
elif anim_args.animation_mode == 'Video Input':
    args.use_init = True

# Clean up unused memory and empty CUDA cache
gc.collect()
torch.cuda.empty_cache()

# Dispatch to appropriate renderer based on animation mode
#These are probably imported from stable diffusion

#lets try to place it within an infer function
def infer(args, anim_args, animation_prompts, root, prompts):
    #render animation (the main one)
    if anim_args.animation_mode == '2D' or anim_args.animation_mode == '3D':
        render_animation(args, anim_args, animation_prompts, root)
    
    #render input video
    elif anim_args.animation_mode == 'Video Input':
        render_input_video(args, anim_args, animation_prompts, root)
    
    #render interpolation
    elif anim_args.animation_mode == 'Interpolation':
        render_interpolation(args, anim_args, animation_prompts, root)
    
    #render image batch 
    else:
        render_image_batch(args, prompts, root)

infer(args, anim_args, animation_prompts, root, prompts)

#———————————————————— 4.2. Launch? ————————————————————————
block.queue(concurrency_count=80, max_size=100).launch(max_threads=150) #from SD to balance/limit requests

with block:
    with gr.Group():
        btn.click(infer, inputs=[image_input], outputs=[video_output, community_icon, loading_icon, share_button])
block.launch()#share=True)



#———————————————————— 4.2. Create Videos from Frames ————————————————————————

#set FPS / video speed
#skip_video_for_run_all = False #@param {type: 'boolean'}
fps = 12 #@param {type:"number"} #HARDCODED FPS HERE: CONSIDER GIVING OPTION TO USERS 

#manual settings for paths

#@markdown **Manual Settings**
use_manual_settings = False #@param {type:"boolean"}  #MOD THIS?
image_path = "/content/drive/MyDrive/AI/StableDiffusion/2023-02/Test14/0_%05d.png" #@param {type:"string"}  #MOD THIS?
mp4_path = "/content/drive/MyDrive/AI/StableDiffusion/2023-02/Test14/0_%05d.mp4" #@param {type:"string"}  #MOD THIS?
render_steps = False  #@param {type: 'boolean'}
path_name_modifier = "x0_pred" #@param ["x0_pred","x"]
make_gif = False
bitdepth_extension = "exr" if args.bit_depth_output == 32 else "png"

# render steps from a single image
if render_steps: 
    
  # get file name and directory of latest output directory
  fname = f"{path_name_modifier}_%05d.png" #MOD THIS?
  all_step_dirs = [os.path.join(args.outdir, d) for d in os.listdir(args.outdir) if os.path.isdir(os.path.join(args.outdir,d))]
  newest_dir = max(all_step_dirs, key=os.path.getmtime)
    
  # create image and video paths
  image_path = os.path.join(newest_dir, fname)
  mp4_path = os.path.join(newest_dir, f"{args.timestring}_{path_name_modifier}.mp4")
  max_frames = str(args.steps)

# render images for a video
else: 
  # create image and video paths with timestamp and bit depth extension
  image_path = os.path.join(args.outdir, f"{args.timestring}_%05d.{bitdepth_extension}")
  mp4_path = os.path.join(args.outdir, f"{args.timestring}.mp4")
  max_frames = str(anim_args.max_frames)

#-------
    
# make video
# create a list with the command and its parameters to call ffmpeg to encode images into an mp4 video
cmd = [
    'ffmpeg',  # specify the name of the executable command
    '-y',  # overwrite output files without asking
    '-vcodec', bitdepth_extension,  # specify the video codec to be used
    '-r', str(fps),  # specify the frames per second (fps) of the output video
    '-start_number', str(0),  # specify the starting number of the image sequence
    '-i', image_path,  # specify the input image sequence (using format specifier)
    '-frames:v', max_frames,  # specify the number of frames to be encoded
    '-c:v', 'libx264',  # specify the video codec to be used for encoding
    '-vf',
    f'fps={fps}',  # specify the fps of the output video again
    '-pix_fmt', 'yuv420p',  # specify the pixel format of the output video
    '-crf', '17',  # specify the constant rate factor (crf) for video quality
    '-preset', 'veryfast',  # specify the encoding speed preset
    '-pattern_type', 'sequence',  # specify the type of pattern used for input file names
    mp4_path  # specify the output mp4 video file path and name
]

# call the ffmpeg command using subprocess to encode images into an mp4 video
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

if process.returncode != 0:  # if ffmpeg command execution returns non-zero code, indicating an error
    # print the error message and raise an exception
    #print(stderr)
    raise RuntimeError(stderr)


#display video
mp4 = open(mp4_path,'rb').read()
#data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
#display.display(display.HTML(f'<video controls loop><source src="{data_url}" type="video/mp4"></video>') )

#make gif
if make_gif:
    gif_path = os.path.splitext(mp4_path)[0]+'.gif'
    cmd_gif = [
        'ffmpeg',
        '-y',
        '-i', mp4_path,
        '-r', str(fps),
        gif_path
    ]
    process_gif = subprocess.Popen(cmd_gif, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


#————————————————————5.1 Add Examples to interface ————————————————————————

#Add examples: see line 158 and 307 at https://huggingface.co/spaces/stabilityai/stable-diffusion/blob/main/app.py





#—————————————————————————————


#the function fn can be either
    #render_animation,
    #render_input_video(args, anim_args, animation_prompts, root),
    #render_interpolation,
    #render_image_batch
    #depending on animation_mode (268)

#the output will be in the variable 'mp4' (or in 'mp4_path' , see 609)

#—————————————————Launch Demo—————————————————————————————
demo = gr.Interface(fn= infer, inputs=image_input, outputs=mp4, title=title, description=description, article=article)
demo.launch()
#demo.launch(auth = ("demo", "demo"), auth_message = "Enter username and password to access application")
