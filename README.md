# Heartbeat Audio Classification
## Introduction:

Heart disease continues to be the world's greatest cause of mortality, accounting for more
deaths each year than all other medical conditions [World Health Organization, 2017]. Because
there is a severe scarcity of doctors in rural areas, 75% of primary care is given by unqualified
practitioners, contributing to the high death rate from cardiovascular diseases (CVDs) in these
areas. Although echocardiograms and electrocardiograms (ECGs) are useful instruments for
keeping an eye on heart health, their high cost and need for specialized training prevent them
from being widely used in environments with limited resources. Our goal is to develop a
dependable, quick, and inexpensive solution that frontline healthcare professionals without
training or anybody with internet connection may use.
The purpose of this approach is to identify people who require additional medical evaluation,
particularly in areas where access to healthcare providers is restricted. It is possible to
considerably lower the risk factors linked to these deaths by detecting CVDs early. In this study,
we automate cardiac auscultation—the process of identifying irregularities in heart sounds. We
introduce an automated heart sound categorization system that combines a deep convolutional
neural network (CNN) with time-frequency heat map representations.

## Our Project

This project aims to classify heartbeat audio recordings into two categories: normal and
abnormal.

1. Normal: healthy heart sounds classified by hearing the pattern lub-dub, lub-dub…
2. Abnormal : heart sounds like murmur, extrasystole, extrahls or an artifact sound
containing no heart sound

The dataset used consists of audio files, which are pre-processed and then used to train a
Convolutional Neural Network (CNN) model for classification.
Using our source, we obtained 2 audios classified as normal, and 7 audios classified as
abnormal in .m4a format, which were converted to .wav format and then later segmented into 60
audio files each.

For further details on the project, kindly refer to the pdf on the link below:

[🔗 Heartbeat Audio Classification pdf](https://github.com/syedmohiuddinzia/mitralValveAbnormalityDetection/blob/main/Heartbeat%20Classification.pdf)
