📊 Project Overview
This project performs sentiment analysis on cleaned comments from various projects using the cardiffnlp/twitter-roberta-base-sentiment model. The system processes comment data, analyzes sentiment patterns, and generates insights about project reception based on user feedback.

🚀 Features
Advanced Sentiment Analysis: Uses state-of-the-art RoBERTa model for accurate sentiment classification

Text Chunking: Handles long comments by splitting them into manageable chunks (max 512 tokens)

Project Comparison: Calculates average sentiment scores across multiple projects

Visual Analytics: Generates visualizations for positive/negative comment distributions

Export Capabilities: Saves results to CSV files for further analysis

📁 Project Structure
text
sentiment-analysis/
├── sentiment-final.ipynb          # Main analysis notebook
├── sentiment_result.csv           # Raw sentiment results
├── average_sentiment_scores.csv   # Aggregated sentiment scores
├── sorted_average_sentiment_scores.csv  # Sorted sentiment rankings
└── README.md
🛠️ Installation & Setup
Prerequisites
Python 3.10+

Jupyter Notebook

NVIDIA GPU (recommended for faster processing)

Dependencies
Install required packages:

bash
pip install pandas numpy seaborn matplotlib transformers torch
Data Requirements
The notebook expects an Excel file (comments_cleaned_sorted.xlsx) with the following structure:

Sr. No.: Serial number

Category: Project category

Project Name: Name of the project

cleaned comments: Preprocessed comment text

🎯 Usage
Running the Analysis
Open the Notebook:

bash
jupyter notebook sentiment-final.ipynb
Data Preparation:

Ensure your Excel file is in the correct path

The notebook will automatically detect and load the data

Execute Cells:

Run cells sequentially to perform:

Data loading and inspection

Model initialization (RoBERTa)

Sentiment analysis processing

Results aggregation and visualization

Data export

Key Outputs
The analysis generates:

Sentiment Results: Detailed sentiment labels and confidence scores for each comment chunk

Average Scores: Mean sentiment scores per project

Rankings: Projects sorted by sentiment score

Visualizations:

Top projects by positive/negative comments

Horizontal bar charts showing sentiment distributions

📈 Results Interpretation
Sentiment Labels
LABEL_0: Negative sentiment

LABEL_1: Neutral sentiment

LABEL_2: Positive sentiment

Sample Output
text
Top 3 Projects by Average Sentiment Score:
1. Thrown into a Deathmatch with a Final Boss! (0.993)
2. OPAL - Organizer (0.993)
3. SCRD Tactical Gear (0.993)
🔧 Customization
Model Selection
You can modify the sentiment analysis model by changing:

python
sentiment_pipeline = pipeline("sentiment-analysis", model="your-model-name")
Text Chunking
Adjust chunk size in the tokenization parameters:

python
max_token_length = 512  # Modify as needed
⚠️ Error Handling
The notebook includes error handling for:

Non-string comment values

Long text processing issues

Model loading failures

Common issues and solutions:

TypeError with float values: Ensure all comments are string type

Memory issues: Reduce chunk size or use GPU acceleration

Model download failures: Check internet connection and Hugging Face access

📊 Performance Considerations
Processing Time: Varies with dataset size (approx. 10-15 minutes for 1,000+ comments)

Memory Usage: RoBERTa model requires significant RAM/VRAM

Optimization: Uses token-based chunking to handle long sequences efficiently

🎓 Methodology
Technical Approach
Text Preprocessing: Combines and chunks comments for model compatibility

Model Inference: Uses RoBERTa for sentiment classification

Aggregation: Calculates weighted averages for project-level sentiment

Visualization: Creates comparative charts using seaborn/matplotlib

Model Details
Base Model: cardiffnlp/twitter-roberta-base-sentiment

Architecture: RoBERTa (Robustly Optimized BERT Pretraining Approach)

Training Data: Twitter social media data

Output: 3-class sentiment classification

📋 Future Enhancements
Potential improvements:

Fine-tuning on domain-specific data

Aspect-based sentiment analysis

Real-time sentiment monitoring

Multi-language support

Interactive dashboards

🤝 Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for:

Bug fixes

Feature additions

Performance improvements

Documentation enhancements

📄 License
This project is open source and available under the MIT License.

🙏 Acknowledgments
Hugging Face for the Transformers library

Cardiff NLP for the pre-trained RoBERTa model

Kaggle community for dataset inspiration

Note: This project is for educational and research purposes. Always ensure proper data privacy and usage rights when working with comment data.



