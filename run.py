from captcha import Captcha
from recognizer import Recognizer

captcha = Captcha()
captcha.split_all_captchas()

recognizer = Recognizer()
recognizer.load_captcha_folder('./letter_imgs')
recognizer.train_model()
recognizer.predict_model('476802131.png')

recognizer.run_in_test_folder('./captcha_test_imgs')
