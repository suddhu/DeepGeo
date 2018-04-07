# DeepGeo

Image Geolocation via Deep Neural Network. 10-701 Course Project. 

## Running 

To acquire 10K images from each state with streetview API (You need to add your streetview API to `streetview_API_key`) - 

```
cd scraper_code
python image_scraper start_state end_state
```

`start_state` and `end_state` can be obtained from `sampler/state_labels.txt`. Scraping data will take significant time. Keep in mind the 25,000 request limit per day. If the limit is crossed, the script will terminate automatically with a message. Each folder in `images` is a state, with an `info.txt` that gives the lat/lng of each image. 


## Plotting

We can plot locations of images obtained from the `info.txt` file by - 
```
cd test
python geo_plotter
```

Will take time...

## Built With

* [Street View API](https://github.com/robolyst/streetview) - Light module for downloading photos from Google street view. 

## Authors

**Sudharshan Suresh**, **Montiel Abello** and **Nathaniel Chodosh**
