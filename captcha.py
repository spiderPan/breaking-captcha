import os
import cv2
import glob
import imutils


class Captcha:
    def __init__(self):
        self.captcha_image_folder = 'captcha_imgs'
        self.letter_folder = 'letter_imgs'
        self.captcha_result_file = self.captcha_image_folder + '/results.txt'

    def resize_to_fit(self, image, width, height):

        (h, w) = image.shape[:2]

        if w > h:
            image = imutils.resize(image, width=width)
        else:
            image = imutils.resize(image, height=height)

        padW = int((width - image.shape[1]) / 2.0)
        padH = int((height - image.shape[0]) / 2.0)

        image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
                                   cv2.BORDER_REPLICATE)
        image = cv2.resize(image, (width, height))

        return image

    def split_captcha_into_letters(self):
        captcha_image_files = glob.glob(os.path.join(self.captcha_image_folder, "*"))
        counts = {}
        for (i, captcha_img_file) in enumerate(captcha_image_files):
            print('[Info] processing image {}/{}'.format(i + 1, len(captcha_image_files)))
            file_name = os.path.basename(captcha_img_file)
            captcha_correct_text, file_type = os.path.splitext(file_name)

            if file_type != '.png':
                continue

            # enhance contrast
            image = cv2.imread(captcha_img_file)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if imutils.is_cv2() else contours[1]

            # split into four letters by bounding
            letter_image_regions = []
            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)

                if w / h > 1.25:
                    half_width = int(w / 2)
                    letter_image_regions.append((x, y, half_width, h))
                    letter_image_regions.append((x + half_width, y, half_width, h))
                else:
                    letter_image_regions.append((x, y, w, h))

            if len(letter_image_regions) != 4:
                continue

            letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

            for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
                x, y, w, h = letter_bounding_box
                letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]
                save_path = os.path.join(self.letter_folder, letter_text)

                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                count = counts.get(letter_text, 1)
                p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
                cv2.imwrite(p, letter_image)

                counts[letter_text] = count + 1

    def rename_captcha_imgs(self):
        with open(self.captcha_result_file) as r:
            line_number = 0
            for captcha in r:
                captcha_data = captcha.strip()
                source_name = self.captcha_image_folder + '/train_' + str(line_number) + '.png'
                new_name = self.captcha_image_folder + '/' + captcha_data + '.png'
                if os.path.isfile(source_name) and not os.path.isfile(new_name):
                    os.rename(source_name, new_name)
                    print('Image train_{}.png ====> {}.png'.format(line_number, captcha_data))
                line_number += 1
