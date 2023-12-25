# This is a sample Python script.
from carbgremover.external_model import remove_background_external, plot_image
from carbgremover.internal_model import remove_background_internal,plot_image
import cv2
from PIL import Image
import requests
from tqdm import tqdm  # Import tqdm for progress bar

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    res = remove_background_external('carbgremover/images/car1.jpg')

    plot_image(res, figsize=(15, 15))
    #res = remove_background_internal('car2.jpg')
    #plot_image(res, figsize=(15, 15))
    # Convert the modified NumPy array back to EagerTensor
    # res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    # import cv2
    # r = cv2.imwrite('rescar2.jpg', res)
    checkpoint_path = "./carbgremover/pretrained_models/sam_vit_h_4b8939.pth"

    # Use Hugging Face API to download the model checkpoint
    url = "https://huggingface.co/spaces/abhishek/StableSAM/resolve/main/sam_vit_h_4b8939.pth"
    response = requests.get(url, stream=True)
    # Save the downloaded checkpoint
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    # Initialize the tqdm progress bar
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    # Save the downloaded checkpoint
    with open(checkpoint_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    # Close the progress bar
    progress_bar.close()

