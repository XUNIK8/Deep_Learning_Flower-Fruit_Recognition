from selenium import webdriver
import time
import urllib.request
import os
from selenium.webdriver.common.keys import Keys

#需要安装Chrome浏览器
# 需要下载Chromedriver 下载地址：http://chromedriver.storage.googleapis.com/index.html
# Chromedriver的版本号要与Chrome要对应

Chromedriver_path="D:/ChromeDriver/chromedriver_win32/chromedriver"# 下载Chromedriver后的地址

name="barbeton daisy"
kinds=0 #1代表水果  0代表花卉

num=1000 #最大图片张数
pages=25# 网页翻页次数，就相当于下拉
def get_img_from_google(Chromedriver_path,name,num,pages,kinds):
    chrome_options  = webdriver.ChromeOptions()

#chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')

#chrome_options.add_argument('--no-sandbox') # 以根用户打身份运行Chrome，使用-no-sandbox标记重新运行Chrome,禁止沙箱启动
    if kinds==1:
        key_words = name + " " + "fruit"
    elif kinds==0:
        key_words=name+" "+"flower"

    browser=webdriver.Chrome(options=chrome_options,executable_path=Chromedriver_path)

    browser.get("https://www.google.com")
    search = browser.find_element_by_name("q")
    search.send_keys(key_words,Keys.ENTER)
    elem = browser.find_element_by_link_text('图片')
    elem.get_attribute('href')
    elem.click()
    value = 0
    for i in range(pages):
        browser.execute_script('scrollBy('+ str(value) +',+1000);')
        value += 1000
        time.sleep(3)

    elem1 = browser.find_element_by_id('islmp')
    sub = elem1.find_elements_by_tag_name('img')

    count = 0

    for i in sub:
        if count > num:
            break

        src = i.get_attribute('src')
        try:
            if src != None:
                src  = str(src)
                print(src)
                count+=1
                urllib.request.urlretrieve(src, os.path.join('E:/Dataset/barbeton daisy',name+str(count)+'.jpg'))
                time.sleep(0.5)
            else:
                raise TypeError
        except TypeError:
            print('fail')


    time.sleep(3)

    browser.quit()
    print("爬取完成，共获得"+str(count)+"张图片")
    return


get_img_from_google(Chromedriver_path,name,num,pages,kinds)