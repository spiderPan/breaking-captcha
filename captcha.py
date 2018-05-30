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

    def split_all_captchas(self):
        captcha_image_files = glob.glob(os.path.join(self.captcha_image_folder, "*"))
        counts = {}
        for (i, captcha_img_file) in enumerate(captcha_image_files):
            print('[Info] processing image {}/{}'.format(i + 1, len(captcha_image_files)))
            file_name = os.path.basename(captcha_img_file)
            captcha_correct_text, file_type = os.path.splitext(file_name)
            if file_type != '.png':
                continue
            image = cv2.imread(captcha_img_file)
            letter_image_regions, image = self.split_captcha_into_letters(image)

            for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
                x, y, w, h = letter_bounding_box
                letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]
                save_path = os.path.join(self.letter_folder, letter_text)

                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                count = counts.get(letter_text, 1)
                p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
                cv2.imwrite(p, letter_image)

                counts[letter_text] = count + 1

    def split_captcha_into_letters(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.copyMakeBorder(image, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if imutils.is_cv2() else contours[1]

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
            print('image is not containing four letters')
            self.print_captcha_img(letter_image_regions, image)

            return [], image

        return sorted(letter_image_regions, key=lambda x: x[0]), image

    def print_captcha_img(self, letter_image_regions, image):
        output = cv2.merge([image] * 3)
        for letter_bounding_box in letter_image_regions:
            x, y, w, h = letter_bounding_box
            cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
            # cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imshow("Output", output)
        cv2.waitKey()

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
