# icAN

Style transfer is an important area of research in the field of generative AI, with applications in a wide range of fields, including art, design, and advertising. This work explores the use of GAN in style transfer for icons across different platform styles.

## Code Structure

The project structure is as follows:

* `ckpt`: Directory to store the trained model checkpoints.

  * `model_demo.pt`: Trained model checkpoint for demo.
* `dataset.py`: Python script for handling the dataset.
* `data/`: Directory containing the training. 
  * `raw`: Raw RGB images for icons.
  * `edge`: Edges for icons.
  * `meta`: Information about icons including reference icons in dataset.
* `main.py`: Main Python script for running the project.
* `models/`: Directory containing models for this project.
  * `basic`: The baseline model for this project.
  * `resnet`: The model using ResNet structure.
* `requirements.txt`: File listing the project's dependencies.
* `scrawler/` : Directory containing utility functions for data preparation.
  * `get_icon_names.py`:  A script that can automate the retrieval of icon names.
  * `download.py`: A script that can download data from [https://icons8.com](https://icons8.com/icon) according to the given icon names list.
  * `download_multi.py`: A script that can download data in multi-threading mode.
  * `preprocess.py`: A script that can preprocess downloading data.
* `eval`: Directory containing evaluation during training

## Usage

1. Ensure that you have the required dependencies installed by running:
``
pip install -r requirements.txt
``
2. Interact with the trained model using `demo.ipynb`.

3. Run `main.py` to train a new model.