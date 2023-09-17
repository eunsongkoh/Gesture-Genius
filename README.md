# Gesture Genius

there is a file that is git ignored: geniusgesture.json this is the authentication key json file

-------------------

## About
"Gesture Genius" stands as a practical solution aimed at eradicating communication barriers, fostering inclusivity, and nurturing connections with the deaf and hard of hearing community. Born in under 36 hours at Hack the North 2023, our ASL (American Sign Language) AI Recognition website is the beginning of a change in technology and communication. Whether you're a part of the ASL community or simply curious about sign language, "Gesture Genius" is here to break down barriers and create meaningful connections. 

## Inspiration
In light of the absence of easily accessible real-time ASL (American Sign Language) translator applications, our team embarked on a mission. Witnessing an era of the rapid influx of AI technology, we recognized the potential to level the playing field, providing equal opportunities for everyone to foster connections and build relationships.

## What it does
Our website gains access to your phone or desktop camera in order to recognize your hand gestures. Using the machine model we trained, our AI should be able to translate ASL into English for people to understand.
Thanks to Google Cloud's Text to Speech API, whichever gestures you sign is instantly converted into words spoken out loud. It's like having a personal interpreter right at your fingertips.

## How we built it
We manually trained our ASL translator AI using Teachable Machine and [this dataset](https://www.kaggle.com/datasets/ayuraj/asl-dataset) from Kaggle that has around 3000 photos per alphabet letter in varying angles. Utilizing JS and HTML/CSS, we created a web application that can be accessed from both mobile and desktop devices. 

## Challenges we ran into
Initiating this project posed a significant set of challenges. Initially, our vision was set on the creation of real-time ASL (American Sign Language) translation glasses. However, as we delved into the complexities of the task, we recognized the impracticality of such an expansive goal within our constrained timeframe and provided outside resources.

In light of this, we opted for a more pragmatic approach—seeking an appropriate model for our project. Yet, this endeavor proved to be far from straightforward. Our research led us through numerous resources, often resulting in encounters with outdated material, misaligned tools that conflicted with our vision, or those exceeding our proficiency.

Additionally, the integration of the Google Cloud Text-to-Speech API into our website presented its own intricate challenges. Establishing a seamless connection with our front-end proved to be a difficult step. Despite these hurdles, our commitment to the project remained steadfast, driving us to navigate through these intricate phases of development.

## Accomplishments that we're proud of
We're proud to highlight several accomplishments achieved during the development of 'Gesture Genius.' We successfully trained our own AI model after many failed attempts. Throughout the project's journey, we demonstrated perseverance, ensuring its successful completion. 'Gesture Genius' represents a solution to a real world problem, being among the first accessible projects of its kind. Our greatest achievement lies in our team's ability to learn and effectively utilize AI models, despite the majority of us having no prior experience in this domain. These accomplishments reflect our commitment to innovation and our dedication to making a tangible difference in technology and communication.

## What we learned
Our journey with 'Gesture Genius' was a rich learning experience that shaped our technical skills and teamwork. We gained proficiency in bridging the gap between backend and frontend development while navigating and learning Google Cloud API. We adeptly learned to utilize AI models and seamlessly incorporate them into websites, including the crucial skill of model training. Despite encountering several challenges, we were able to work together to quickly bring other solutions to the table. These newfound skills and knowledge not only contributed to the success of our project but also provided us with a strong foundation for future projects. 

## What's next for Gesture Genius: ASL AI Recognition
Looking ahead for 'Gesture Genius: ASL AI Recognition,' our vision is to continue refining and expanding our capabilities. We aim to enhance the accuracy of our AI models through further training, incorporating diverse backgrounds to ensure robust performance. Additionally, we aspire to implement word recognition capabilities, eliminating the need for finger spelling and enhancing the fluidity of communication. There are also plans to implement a real time text box for translations to easily follow with conversations. Our commitment to breaking down communication barriers remains unwavering, and we are excited to embark on the next phase of this transformative journey.
