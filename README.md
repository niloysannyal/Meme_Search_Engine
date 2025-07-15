# 🎯 Meme Search Engine 🔍

A powerful meme search engine that allows users to find similar memes using text queries or image uploads.  
Built with **CLIP (Contrastive Language–Image Pre-training)**, **PyTorch**, and **Streamlit** for a smooth and interactive web experience.


---


## 🌐 Web App

<p align="center">
  <img src="https://github.com/user-attachments/assets/1eafc01a-0704-4530-8ef8-8881f223f81c" alt="Meme Search Engine" width="1000" />
</p>

<p align="center">
  <a href="https://memesearchengine.streamlit.app/">
    <img src="https://img.shields.io/badge/LIVE-VISIT%20NOW-blue?style=for-the-badge&logo=streamlit" alt="Live">
  </a>
</p>


---

## 🚀 Features

- Search memes by text query with sentiment filtering (Any, Positive, Neutral, Negative).
- Upload an image to find visually similar memes.
- Precomputed embeddings for fast search performance.
- Download memes directly from search results.
- Responsive and user-friendly web interface powered by Streamlit.

---

## 📁 Repository Structure
```
Meme_Search_Engine/
├── Memes/                   # Dataset images and embeddings
│   ├── embeddings.pkl       # Precomputed image embeddings
│   ├── labels/              # CSV labels and metadata
│       ├── labels.csv
├── src/
│   ├── embed_images.py      # Script to create embeddings
│   ├── inference.py         # MemeSearch class for searching
│   └── config.py            # Configuration constants and paths
├── app.py                   # Streamlit web app entrypoint
├── requirements.txt         # Python dependencies
└── README.md                # This file
```


## 🛠️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/NiloySannyal/Meme_Search_Engine.git
   cd Meme_Search_Engine
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/MacOS
   venv\Scripts\activate         # Windows
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Prepare the dataset:
   - Ensure your Memes/ folder contains meme images and the corresponding labels.csv inside Memes/labels/.
5. Generate embeddings (if needed):
   ```bash
    python src/embed_images.py
6. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## ⚙️ Configuration
  Edit the src/config.py file to set your:
- Dataset directory (DATA_DIR)
- Label CSV path (LABEL_CSV)
- Model name (MODEL_NAME)
- Sentiment options (SENTIMENT_OPTIONS)


## 🔍 Usage
- Use the sidebar to enter a text query or upload an image.
- Filter results by sentiment.
- Select how many top results to display.
- Click the download link under each result to save memes locally.


## 📦 Dependencies
- torch
- torchvision
- Pillow
- tqdm
- pandas
- streamlit
- [CLIP](https://github.com/openai/CLIP) (OpenAI's official CLIP repository)


## 🙌 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Niloy Sannyal**  
GitHub: [niloysannyal](https://github.com/niloysannyal)  
Email: [niloysannyal@gmail.com](mailto:niloysannyal@gmail.com)


  
