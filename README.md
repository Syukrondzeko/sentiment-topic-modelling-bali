# Multimodal Sentiment Analysis and Topic Modelling of Bali Tourism Reviews

This project analyses Google Maps reviews of tourist attractions in Bali. It uses both the review text and the photos that reviewers attached.

- **Sentiment classification**: predicts negative, neutral or positive sentiment from the star rating. It compares four inputs: review text plus an image caption, text only, caption only, and CLIP image embeddings.
- **Topic modelling**: finds recurring themes in the reviews with BERTopic, then merges them into interpretable tourism themes.

## Pipeline

Everything is in [`Notebook.ipynb`](Notebook.ipynb):

| Step | What it does | Output |
|---|---|---|
| 1. Compile dataset | Merges the scraped CSV exports and keeps reviews that have English text | `Dataset.csv` |
| 2. Download images | Downloads the first photo of each review | `images/` |
| 3. Caption images | Generates a caption for each photo with [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) | `captions/` |
| 4. Clean dataset | Attaches the captions and drops rows without text or caption | `Dataset_with_captions.csv`, `Dataset_clean.csv` |
| 5. Sentiment classification | Fine-tunes `bert-base-uncased` on text and/or captions, and trains an MLP on CLIP ViT-B/32 image embeddings | Printed metrics and a comparison table |
| 6. Topic modelling | Runs BERTopic with `all-MiniLM-L6-v2` embeddings, choosing the number of topics by silhouette score | `bertopic_result.csv`, `bertopic_topic_info.csv` |

Sentiment labels come from the star ratings: 1–2 stars is negative, 3 is neutral, 4–5 is positive. The positive class is downsampled so the classes are balanced. The data is split 70/15/15 with stratification, and every experiment uses the same split.

## Data

The reviews were collected with the Apify *Google Maps Reviews Scraper*. Text from reviews not written in English uses the translation Google provides (`textTranslated`).

The dataset is not included in this repository. That covers the raw scraper exports, the processed CSVs, the review photos and the captions. The raw exports contain reviewer identifiers and profile links, and the photos belong to the reviewers. To reproduce the results, put the raw exports in `raw_datasets/` and run the notebook from step 1.

## Running

The notebook runs on Google Colab with a GPU, or locally:

```bash
pip install -r requirements.txt
jupyter notebook Notebook.ipynb
```

All paths are relative to `DATA_DIR`. On Colab, `DATA_DIR` defaults to `/content/drive/MyDrive/Multimodal Bali` and Google Drive is mounted automatically. Locally, it defaults to the repository folder. To use another location, set the `DATA_DIR` environment variable.

If you already have the processed CSVs from steps 1–4, you can skip straight to step 5.

## Results

Rerun the notebook to get the sentiment comparison table (test accuracy and macro F1 per input type) and the topic list. Topic IDs change each time the model is refit, so check the "Topic interpretation" table in the notebook against the newly printed topics.
