# Violin recognition model

Using **deep learning techniques** usually used for **face recognition** (i.e. Arcface and triplet loss), we developed a model that can take any violin picture and match it to violins from an existing database. 

We undertook this project for Tarisio, an auction house for fine stringed instruments. We used Tario's database Cozio, described on the website as having pictures of "over 36,000 individual instruments and bows by over 3,500 makers."


Below is a video of an app showing the typical use case. This app is coded with dash and powered with AWS lambda functions.

https://user-images.githubusercontent.com/85825309/236393802-0df2164b-bce4-4935-ba97-5caac1539600.mp4

Users can take a picture of a violin's back and drop it in the left frame.
A first model (Facebook's detectron2) detects **keypoints** on the violin. A program then uses these keypoints to 1. **center and rotate** the violin and 2. **crop** a rectangle inside the violin's back. 
A second model, trained using face-recognition training methods, compares the wood pattern inside this rectangle to all images in the dataset. It then returns the N violin back pictures with the most similar wood patterns. In the videos, the correct violin comes up first, and "lookalikes" come up after.

The following videos show that the model works even with bright lights flashing on the violin's varnished wood, and with the violin rotated 90 degrees. All in all, the recognition model achieved a **rank-1 accuracy** of 93% and a **rank-5 accuracy** of 96% on both a validation and a test dataset of over 1700 images each.

https://user-images.githubusercontent.com/85825309/236393212-dff37932-8996-4778-ab9e-df882c0eeb74.mp4



https://user-images.githubusercontent.com/85825309/236393274-32aa9f48-7b7a-414d-8a4f-6b44923eaddc.mp4

