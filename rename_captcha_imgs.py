import os

CAPTCHA_IMAGE_FOLDER = 'captcha_imgs'
CAPTCHA_RESULTS = CAPTCHA_IMAGE_FOLDER + '/results.txt'

with open(CAPTCHA_RESULTS) as r:
    line_number = 0
    for captcha in r:
        captcha_data = captcha.strip()
        source_name = CAPTCHA_IMAGE_FOLDER + '/train_' + str(line_number) + '.png'
        new_name = CAPTCHA_IMAGE_FOLDER + '/' + captcha_data + '.png'
        if os.path.isfile(source_name) and not os.path.isfile(new_name):
            os.rename(source_name, new_name)
            print('Image train_{}.png ====> {}.png'.format(line_number, captcha_data))
        line_number += 1
