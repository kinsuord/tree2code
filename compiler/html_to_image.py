from selenium import webdriver
import json 
import os
from PIL import Image
from io import BytesIO

# options = webdriver.ChromeOptions()
# options.add_argument("--window-size=1920,1080")
# # options.add_argument('headless')

# driver=webdriver.Chrome("./chromedriver.exe", options = options)

# f= open("./1.html","r")
# html_content = f.read()

# driver.get("data:text/html;charset=utf-8," + html_content)
# driver.execute_script("""
#   document.location = 'about:blank';
#   document.open();
#   document.write(arguments[0]);
#   """, html_content)

# # driver.execute_script("document.write('{}')".format(json.dumps(html_content)))
# # driver.get("file:///" + os.getcwd() + "//" + '0.html')

# element = driver.find_element_by_xpath("//main") # find part of the page you want image of
# location = element.location
# size = element.size

# img = driver.get_screenshot_as_png()
# img = Image.open(BytesIO(img))
# left = location['x']
# top = location['y']
# right = location['x'] + size['width']
# bottom = location['y'] + size['height']

# img = img.crop((left, top, right, bottom)) # defines crop points
# img.save('0.png') 


class Driver():
    def __init__(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--window-size=1920,1080")
        options.add_argument('headless')
        self.driver = webdriver.Chrome("./compiler/chromedriver.exe", options = options)

    def html_to_img(self, html_content, saveToFileName = None):
        '''
        Render html code to image.

        Input: 
            html_content: html string content
            saveToFileName: String. If provided, result will save to file
        Output:
            return image array
        '''
        self.driver.get("data:text/html;charset=utf-8," + html_content)

        self.driver.execute_script("""
        document.location = 'about:blank';
        document.open();
        document.write(arguments[0]);
        """, html_content)

        img = self.driver.get_screenshot_as_png()
        img = Image.open(BytesIO(img))

        element = self.driver.find_element_by_class_name('container')
        location = element.location
        size = element.size

        left = location['x']
        top = location['y']
        right = location['x'] + size['width']
        bottom = location['y'] + size['height']

        # print(left, top, right, bottom)
   
        img = img.crop((left, top, right, bottom)) 
        # img = img.crop((375, 0, 1545, 780)) 

        if saveToFileName != None:
            img.save(saveToFileName) 

        return img

    def htmlfile_to_img(self, filename, saveToFileName = None):
        '''
        Render html file to image.

        Input: 
            filename: html filename
            saveToFileName: String. If provided, result will save to file
        Output:
            return image array
        '''

        f= open(filename,"r")
        html_content = f.read()

        self.driver.get("data:text/html;charset=utf-8," + html_content)

        self.driver.execute_script("""
        document.location = 'about:blank';
        document.open();
        document.write(arguments[0]);
        """, html_content)

        element = self.driver.find_element_by_xpath("//main") 
        location = element.location
        size = element.size

        img = self.driver.get_screenshot_as_png()
        img = Image.open(BytesIO(img))

        left = location['x']
        top = location['y']
        right = location['x'] + size['width']
        bottom = location['y'] + size['height']

        img = img.crop((left, top, right, bottom)) 

        if saveToFileName != None:
            img.save(saveToFileName) 

        return img