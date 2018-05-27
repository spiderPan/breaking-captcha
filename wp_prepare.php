<?php
if (!class_exists('ReallySimpleCaptcha')) {
    echo 'Plugin is not installed!';
    exit;
}

$captcha_instance = new ReallySimpleCaptcha();
$count = 0;
while ($count < 20000) {
    $word = $captcha_instance->generate_random_word();
    $prefix = $count;
    $captcha_instance->generate_image($word, $word);
    $count++;
}
//echo $content;
echo 'generated!';
exit;
