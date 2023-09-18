# Violin recognition model

We have developed a model using **deep learning techniques**, specifically Arcface and triplet loss, which are commonly used for **face recognition**. This model is capable of matching any violin picture with violins from an existing database, even if these violins were not included in the training dataset.

This project was undertaken for [Tarisio](https://tarisio.com/), an auction house specializing in fine stringed instruments. We utilized Tarisio's database, [Cozio](https://tarisio.com/cozio-archive/about-cozio/), which is described on their website as containing pictures of "over 36,000 individual instruments and bows by over 3,500 makers."

Below is a video demonstrating the typical use case of the application. The app has been coded using dash and is powered by AWS lambda functions.

https://user-images.githubusercontent.com/85825309/236393802-0df2164b-bce4-4935-ba97-5caac1539600.mp4

Users can take a picture of a violin's back and drop it in the left frame.
Initially, the [detectron2](https://ai.meta.com/tools/detectron2/) framework developed by Meta was used to fine-tune a **keypoint-rcnn** model (see He and al. [Mask R-CNN](https://arxiv.org/abs/1703.06870), section 5). This model was trained to detect important keypoints on the violin. Subsequently, a program uses these keypoints to accomplish two tasks: 1) centering and rotating the violin, and 2) cropping a rectangle within the back of the violin. 
A second model, which has been trained using face-recognition training methods, compares the wood pattern within this rectangle to all the images in the dataset. It then provides the N violin back pictures that exhibit the most similar wood patterns. In the videos, the correct violin is displayed first, followed by the "lookalikes".

The following videos show that the model works even with bright lights flashing on the violin's varnished wood, and with the violin rotated 90 degrees. Overall, the recognition model achieved a **rank-1 accuracy** of 93% and a **rank-5 accuracy** of 96% on both a validation and a test dataset of over 1700 images each.

https://user-images.githubusercontent.com/85825309/236393212-dff37932-8996-4778-ab9e-df882c0eeb74.mp4



https://user-images.githubusercontent.com/85825309/236393274-32aa9f48-7b7a-414d-8a4f-6b44923eaddc.mp4

