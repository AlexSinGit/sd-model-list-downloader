import os
import gradio as gr
from modules import scripts, shared, sd_models, sd_vae
from modules.ui_extra_networks import extra_networks
import aiohttp
from PIL import Image
from io import BytesIO
import json
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
import asyncio
import hashlib
import threading
import aiofiles


class ModelDownloaderExtension(scripts.Script):
    def __init__(self):
        super().__init__()
        self.base_dir = shared.cmd_opts.data_dir
        self.category_dirs = {
            "model": "models/Stable-diffusion",
            "lora": "models/Lora",
            "embedding": "embeddings",
            "vae": "models/VAE",
        }

    def title(self):
        return "Model Downloader"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Model Downloader", open=False):
            base_dir = gr.Textbox(label="Base Download Directory", value=self.base_dir)
            txt_file = gr.File(label="Upload TXT File", file_types=[".txt"])
            search_box = gr.Textbox(label="Search Models")
            model_list = gr.HTML()
            update_button = gr.Button("Update Model List")
            download_status = gr.HTML(elem_id="download_status")
            progress_bar = gr.HTML(elem_id="progress_bar")

            update_button.click(
                fn=self.update_model_list,
                inputs=[txt_file, base_dir, search_box],
                outputs=[model_list],
            )

            search_box.change(
                fn=self.update_model_list,
                inputs=[txt_file, base_dir, search_box],
                outputs=[model_list],
            )

        return [
            base_dir,
            txt_file,
            search_box,
            model_list,
            update_button,
            download_status,
            progress_bar,
        ]

    def parse_txt_file(self, file_path):
        models = []
        current_model = None
        model_name = ""

        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                if(line.startswith('#') or line.startswith('http')):
                    if (
                        current_model
                        and "url" in current_model
                        and "name" in current_model
                        and "img" in current_model
                    ):
                        models.append(current_model)
                        current_model = {"type": model_name}
                if line.startswith("#"):
                    model_name = line[1:]
                    current_model = {"type": model_name}
                if line.startswith("http"):
                    if current_model:
                        url, name = line.split("|")
                        current_model["url"] = url.strip()
                        current_model["name"] = name.strip()
                if line.startswith(r"//"):
                    if current_model:
                        key, value = line[2:].split(" ", 1)
                        current_model[key] = value

        models.append(current_model)
        return models

    def filter_models(self, models, search_term):
        if not isinstance(search_term, str) or len(search_term) == 0:
            return models
        return [
            model for model in models if search_term.lower() in model["name"].lower()
        ]

    def update_model_list(self, txt_file, base_dir, search_term):
        if not txt_file:
            return "Please upload a TXT file."

        self.base_dir = base_dir
        models = self.parse_txt_file(txt_file.name)
        filtered_models = self.filter_models(models, search_term)
        html = self.generate_model_list_html(filtered_models)
        return html

    def generate_model_list_html(self, models):
        html = "<div style='max-height: 500px; overflow-y: auto;'>"

        # Group models by type
        model_groups = {}
        for model in models:
            model_type = model.get("type", "Unknown")
            if model_type not in model_groups:
                model_groups[model_type] = []
            model_groups[model_type].append(model)

        # Generate HTML for each group
        for model_type, group_models in model_groups.items():
            html += f"<h2>{model_type}</h2>"
            html += "<div style='display: flex; flex-wrap: wrap; gap: 20px;'>"
            for model in group_models:
                html += self.generate_model_card_html(model)
            html += "</div>"

        html += "</div>"
        return html

    def generate_model_card_html(self, model):
        return f"""
        <div style='border: 1px solid #ddd; padding: 10px; width: 300px;'>
            <img src='{model.get("img", "")}' style='width: 100%; min-height: 380px; object-fit: cover;'>
            <h3>{model.get('name', 'Unnamed Model')}</h3>
            <p>Type: {model.get('type', 'Unknown')}</p>
            <a href='{model.get('page', '#')}' target='_blank'>Model Page</a>
            <button onclick='downloadModel("{model.get('url', '')}", "{model.get('img', '')}", "{model.get('trigger', '')}", "{self.base_dir}", "{model.get('name', 'unnamed')}", "{model.get('type', 'Unknown')}")'>Download</button>
        </div>
        """

    async def download_model(
    self, model_url, image_url, trigger_words, base_dir, model_name, model_type
    ):
        category_dir = self.category_dirs.get(model_type.lower(), "other")
        download_dir = os.path.join(base_dir, category_dir)
        os.makedirs(download_dir, exist_ok=True)
        sha256_hash = hashlib.sha256()

        try:
            async with aiohttp.ClientSession() as session:
                # Download model file
                model_file_path = os.path.join(download_dir, model_name)
                temp_file_path = f"{model_file_path}.temp"
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
                }
                if os.path.exists(temp_file_path):
                    file_size = os.path.getsize(temp_file_path)
                    headers['Range'] = f'bytes={file_size}-'

                async with session.get(model_url, headers=headers) as model_response:
                    model_response.raise_for_status()
                    total_size = int(model_response.headers.get("content-length", 0))
                    if 'content-range' in model_response.headers:
                        total_size = int(model_response.headers['content-range'].split('/')[-1])
                    
                    mode = 'ab' if os.path.exists(temp_file_path) else 'wb'
                    downloaded = os.path.getsize(temp_file_path) if os.path.exists(temp_file_path) else 0
                    
                    chunk_size = 1024 * 1024  # 1 MB chunks
                    async with aiofiles.open(temp_file_path, mode) as f:
                        async for chunk in model_response.content.iter_chunked(chunk_size):
                            await f.write(chunk)
                            sha256_hash.update(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                yield json.dumps({"progress": progress})
                            await asyncio.sleep(0)  # Allow other tasks to run

                os.rename(temp_file_path, model_file_path)

                # Download image file (in parallel with model download)
                async def download_image():
                    async with session.get(image_url) as image_response:
                        image_response.raise_for_status()
                        image_data = await image_response.read()
                        image = Image.open(BytesIO(image_data))
                        image = image.convert("RGB")
                        preview_name = os.path.splitext(model_name)[0]
                        image_file_path = os.path.join(
                            download_dir, f"{preview_name}.preview.jpeg"
                        )
                        image.save(image_file_path, "JPEG", quality=95)

                await download_image()

            # Save trigger words
            if trigger_words:
                preview_name = os.path.splitext(model_name)[0]
                metadata = {
                    "description": "",
                    "sd version": "",
                    "modelId": "models",
                    "activation text": trigger_words,
                    "sha256": sha256_hash.hexdigest(),
                }
                json_file_path = os.path.join(
                    download_dir, f"{os.path.splitext(model_name)[0]}.json"
                )
                async with aiofiles.open(json_file_path, "w") as f:
                    await f.write(json.dumps(metadata, indent=2))

            # Load model in a separate thread
            def load_model_thread():
                self.reload_models(model_type)

            thread = threading.Thread(target=load_model_thread)
            thread.start()

            yield json.dumps(
                {
                    "message": f"Model '{model_name}' downloaded successfully to {download_dir}! Loading model in background..."
                }
            )

        except aiohttp.ClientError as e:
            yield json.dumps(
                {"message": f"Failed to download model '{model_name}'. Error: {str(e)}"}
            )
        except Exception as e:
            yield json.dumps(
                {
                    "message": f"An error occurred while downloading model '{model_name}'. Error: {str(e)}"
                }
            )

    async def reload_models(self, model_type):
        try:
            if model_type.lower() == "model":
                sd_models.list_models()
                sd_models.load_model()
            elif model_type.lower() == "vae":
                sd_vae.refresh_vae_list()
            elif model_type.lower() == "lora":
                lora_module.list_available_loras()  # Use the imported lora module
            elif model_type.lower() == "embedding":
                await self.reload_embeddings()
            else:
                print(f"Unknown model type: {model_type}. No reload performed.")

            # Refresh all extra networks
            extra_networks.initialize()
            for extra_network in extra_networks.extra_network_registry:
                extra_network.refresh()

            print(f"Reloaded models for type: {model_type}")
        except Exception as e:
            print(f"Error reloading models: {str(e)}")

def on_app_started(demo: gr.Blocks, app: FastAPI):
    @app.get("/sdapi/v1/download_model")
    async def api_download_model(
        model_url: str,
        image_url: str,
        trigger_words: str,
        download_dir: str,
        model_name: str,
        model_type: str,
    ):
        extension = next(
            ext
            for ext in scripts.scripts_data
            if isinstance(ext.script_class(), ModelDownloaderExtension)
        )

        return EventSourceResponse(
            extension.script_class().download_model(
                model_url,
                image_url,
                trigger_words,
                download_dir,
                model_name,
                model_type,
            )
        )


scripts.script_callbacks.on_app_started(on_app_started)

# JavaScript for the extension
js = """
function downloadModel(modelUrl, imageUrl, triggerWords, downloadDir, modelName, modelType) {
    const statusElement = gradioApp().querySelector('#download_status');
    const progressBarElement = gradioApp().querySelector('#progress_bar');
    
    statusElement.innerHTML = `Downloading ${modelName}...`;
    progressBarElement.innerHTML = `
        <div style="width: 100%; background-color: #ddd;">
            <div id="progress" style="width: 0%; height: 30px; background-color: #4CAF50; text-align: center; line-height: 30px; color: white;">
                0%
            </div>
        </div>
    `;
    
    const eventSource = new EventSource(`/sdapi/v1/download_model?model_url=${encodeURIComponent(modelUrl)}&image_url=${encodeURIComponent(imageUrl)}&trigger_words=${encodeURIComponent(triggerWords)}&download_dir=${encodeURIComponent(downloadDir)}&model_name=${encodeURIComponent(modelName)}&model_type=${encodeURIComponent(modelType)}`);
    
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        if (data.progress !== undefined) {
            const progressElement = gradioApp().querySelector('#progress');
            progressElement.style.width = `${data.progress}%`;
            progressElement.innerHTML = `${data.progress.toFixed(2)}%`;
        }
        if (data.message) {
            statusElement.innerHTML = data.message;
            if (data.message.includes("downloaded successfully") || data.message.includes("Error")) {
                eventSource.close();
            }
        }
    };
    
    eventSource.onerror = function(error) {
        console.error('EventSource failed:', error);
        statusElement.innerHTML = `Error: ${error}`;
        eventSource.close();
    };
}
"""

scripts.script_callbacks.on_ui_settings(
    lambda: shared.opts.add_option(
        "modeldownloader_js",
        shared.OptionInfo(
            js, "Model Downloader JS", section=("Model Downloader", "Model Downloader")
        ),
    )
)
