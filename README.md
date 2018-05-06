# DeepGeo

Image Geolocation via Deep Neural Network. 10-701 Course Project. 


## Setup 

Install the Google Maps package for python 

```
sudo pip install googlemaps
```

Change `images_dir = '/already/created/path/to/image/folder'` and get an API key from [here]{https://developers.google.com/maps/documentation/streetview/get-api-key}. Add that to `streetview_API_key`. 
## Running 

To acquire 10K images from each state with streetview API (You need to add your streetview API to `streetview_API_key`) - 

```
cd scraper_code
python image_scraper start_state end_state
```

`start_state` and `end_state` can be obtained from `sampler/state_labels.txt`. Scraping data will take significant time. Keep in mind the 25,000 request limit per day. If the limit is crossed, the script will terminate automatically with a message. Each folder in `images` is a state, with an `info.txt` that gives the lat/lng of each image. 

## Training

All the code for training models can be found in the resnet directory. The main file is 'train.py', run 'train.py --help' for a description of how to train and evaluate a model. Code for building the networks is found in 'resnet.py'.

## Plotting

We can plot locations of images obtained from the `info.txt` file by - 
```
cd test
python geo_plotter
```


## Troubleshooting 

If you get the error 

```
InsecurePlatformWarning: A true SSLContext object is not
available. This prevents urllib3 from configuring SSL appropriately and 
may cause certain SSL connections to fail. For more information, see 
https://urllib3.readthedocs.org/en/latest  
```

Do this - 
```
sudo pip install 'requests[security]'
```
Will take time...

## Built With

* [Street View API](https://github.com/robolyst/streetview) - Light module for downloading photos from Google street view. 

## Authors

**Sudharshan Suresh**, **Montiel Abello** and **Nathaniel Chodosh**
