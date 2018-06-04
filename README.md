# CNN to break captcha

This is a project that using Machine Learning(CNN model) to break simple captcha. To make it easy, we choose [Really Simple Captcha](https://contactform7.com/captcha/) plugin, one of the most popular WordPress captcha plugin, as example.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them
1. Local WordPress environment to run the plugin
2. [WP CLI](https://wp-cli.org/#installing)
3. Install the plugins by using WP CLI within your WP folder
    ```
    wp plugin install contact-form-7 --activate
    wp plugin install really-simple-captcha --activate
    ```
4. Python 3.*
5. Pip

### Installing

A step by step series of examples that tell you how to get a development env running

1. Clone the repo to your local server
    ```
    git clone git@github.com:spiderPan/breaking-captcha.git
    ```
2. Install requirements packages within the repo
    ```
    cd breaking-captcha
    pip install -r requirments.txt
    ```
3. Move `wp_prepare.php` into your WordPress Theme
    ```
    cp wp_prepare.php LOCAL_WP/wp-content/themes/YOUR_THEME/
    ```
4. Adding the following line to the begin of your WordPress theme's `functions.php`
    ```
    include wp_prepare.php
    ```
5. Start generating captcha images by calling CLI
    ```
    wp
    ```
    When it's done, all 20000 captcha images will be generated in `wp-content/plugins/really-simple-captcha/tmp`
6. Copy all captchas into repo's `captcha_imgs` folder
    ```
    cp -r LOCAL_WP/wp-content/plugins/really-simple-captcha/tmp/*.png breaking-captcha/captcha_imgs/
    ```

## Running the tests

Here is a break down in `run.py`

1. Split captcha images into letters 
    ```
    captcha = Captcha()
    captcha.split_all_captchas()
    ```
2. Train the CNN model
    ```
    recognizer = Recognizer()
    recognizer.load_captcha_folder('./letter_imgs')
    recognizer.train_model()
    ```
3. Make prediction for individual image
    ```
    recognizer.predict_model(IMAGE_FILE)
    ```
4. OR, instead of doing #3, repeat Installing #5 and coping all fresh captcha images into a new testing folder within this repo, for example, `captcha_test_imgs` and then
    ```
    recognizer.run_in_test_folder('./captcha_test_imgs')
    ```

If it's first time running it, just comment out #4 and run `python run.py`

If model already been trained, then comment out #2 and choose one of #3 and #4 to predict

## Future Plan

Currently the CNN model can reach 99% accuracy within 10 epoch. There are two things I'm thinking to work on in the future

1. Since the project is using OpenCV to split letters from captcha images, sometimes captcha file is not correctly split into four letters, which is the bottleneck of recognitions.

2. Automate HTTP request to sending out email through contact form 7 site by using the model to break the captcha site.