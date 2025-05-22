## F1 track layout recognition app 🏎️
A small side-project for GHOST Science Club

### This branch is specifecly created for streamlit app, it does not contain the feedback loop

##### How to run this app localy
1. Install requirments
2. You need to run every notebook in correct order
3. When in project's directory use command below
```bash
streamlit run app.py
```

This will setup the app on localhost and browser should open automatically.

### About project

##### Motivation
As an F1 enthusiast and frequent GeoGuessr player, I wanted to combine my passions into a single AI project. Predicting race outcomes felt overdone, so I set out to build a model that recognizes a Formula 1 circuit from a photo—hoping it would learn subtle visual cues like curbing patterns or surrounding landmarks.

##### Phase 1: Photo-based CNN
I began by gathering images of F1 tracks via a custom web crawler. Unfortunately, most images didn’t match the search queries, and gathering a clean, labeled dataset proved very difficult. After several iterations, the CNN-based classifier on raw track photos failed to reach satisfactory accuracy - so I decided to simplify the problem.

##### Phase 2: Layout-only Recognition
I pivoted to identifying circuits from their layout diagrams. I reused the crawler to fetch overhead layouts, then trained a binary filter to remove non-layout images. Although the filter isn’t perfect, it allowed me to build a workable training set. The resulting CNN achieved strong accuracy on layout diagrams.

##### Phase 3: Web Application
With a reliable layout classifier in hand, I built a lightweight web app featuring:

- A canvas where users can draw their own circuit layout by hand

- Instant prediction of which F1 track it is, based on the drawn lines

- A feedback system asking “Is this the correct track? If not, which one?”

- Logging of every correction so new feedback can later be used to retrain and improve the model

- The model is loaded directly in code from a local PyTorch .pth weights file - no external API required

##### Key Skills & Technologies

- Transfer Learning: PyTorch & Transfer Learning: Fine-tuned a pre-trained CNN and managed .pth weight files

- Web Scraping: Built a crawler to collect thousands of track images and layouts

- Data Cleaning: Implemented an image filter model and manual verification loop to refine the dataset

- Web Development: Created a responsive frontend with a drawing canvas and integrated the local model loading

##### Next Steps

- Manually review and clean the layout dataset to boost classifier accuracy

- Consider deploying the web app to the cloud if there’s enough interest
