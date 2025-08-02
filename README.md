# Sinhala-Tamil Neural Machine Translation Using Pivot-Based Transfer Learning

A Flask web application that provides neural machine translation between Sinhala and Tamil languages using English as a pivot language. This project implements a proof of concept using the mBART model with adapter-based transfer learning for low-resource language pairs.

## 🌟 Features

- **Web-based Translation Interface**: Easy-to-use web interface for translation
- **Pivot-based Translation**: Uses English as an intermediate language for Sinhala ↔ Tamil translation
- **Adapter-based Fine-tuning**: Leverages PEFT (Parameter Efficient Fine-Tuning) with adapters
- **mBART Integration**: Built on Facebook's mBART-large-50-many-to-many-mmt model
- **Real-time Translation**: Instant translation through the web interface

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **ML Framework**: PyTorch
- **NLP Library**: Hugging Face Transformers
- **Model**: mBART-large-50-many-to-many-mmt
- **Fine-tuning**: PEFT (Parameter Efficient Fine-Tuning)
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Gunicorn ready

## 📁 Project Structure

```
sinhala-to-tamil-translator/
├── app.py                          # Flask application
├── requirements.txt                # Python dependencies
├── Procfile                       # Deployment configuration
├── README.md                      # Project documentation
├── models/                        # Trained adapter models
│   ├── english_tamil/            # English-Tamil adapter
│   └── sinhala_english/          # Sinhala-English adapter
├── templates/
│   └── index.html                # Web interface
└── venv_py311/                   # Virtual environment
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Sufficient RAM (8GB+ recommended for model loading)
- CUDA-compatible GPU (optional, for faster inference)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/rajika97/sinhala-to-tamil-translator.git
   cd sinhala-to-tamil-translator
   ```

2. **Set up virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**

   ```bash
   python app.py
   ```

5. **Access the web interface**
   Open your browser and navigate to `http://localhost:5000`

## 🔧 Usage

1. Open the web application in your browser
2. Enter Sinhala text in the input field
3. Click "Translate" to get the Tamil translation
4. The system will automatically:
   - Translate Sinhala to English (using the Sinhala-English adapter)
   - Translate English to Tamil (using the English-Tamil adapter)

## 🏗️ Architecture

The translation system uses a **pivot-based approach**:

```
Sinhala → English → Tamil
```

### Components:

1. **Base Model**: mBART-large-50-many-to-many-mmt
2. **Sinhala-English Adapter**: Fine-tuned for Sinhala to English translation
3. **English-Tamil Adapter**: Fine-tuned for English to Tamil translation
4. **Flask Web Server**: Provides REST API and web interface

## 📊 Model Information

- **Base Model**: `facebook/mbart-large-50-many-to-many-mmt`
- **Fine-tuning Method**: PEFT adapters (LoRA)
- **Languages Supported**: Sinhala (si), English (en), Tamil (ta)
- **Model Size**: ~2.4GB (base model + adapters)

## 🚀 Deployment

The application is configured for deployment with Gunicorn:

```bash
gunicorn app:app
```

The `Procfile` is included for Heroku deployment.

## 📝 Research Context

This project demonstrates the effectiveness of:

- **Transfer Learning** for low-resource language pairs
- **Adapter-based Fine-tuning** for parameter-efficient training
- **Pivot-based Translation** using English as an intermediate language
- **Multi-step Translation Pipeline** for improved accuracy

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Hugging Face for the Transformers library and pre-trained models
- Facebook AI for the mBART model
- The open-source community for various tools and libraries

## 📚 Related Work

This implementation is based on research in neural machine translation for low-resource languages and demonstrates practical applications of:

- Multilingual pre-trained models
- Parameter-efficient fine-tuning
- Pivot-based translation strategies

---

**Note**: This is a proof-of-concept implementation. For production use, consider additional optimizations, error handling, and security measures.
