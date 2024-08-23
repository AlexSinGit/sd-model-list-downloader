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