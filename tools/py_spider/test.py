#!/usr/bin/env python
# -*- coding:utf-8 -*-
from selenium import webdriver
import time
 
options = webdriver.ChromeOptions()
prefs = {
    'profile.default_content_setting_values':
        {
            'notifications': 2
        }
}
#options.add_experimental_option('prefs', prefs)  # 关掉浏览器左上角的通知提示
options.add_argument("disable-infobars")  # 关闭'chrome正受到自动测试软件的控制'提示
driver = webdriver.Chrome(options=options)
driver.maximize_window()
driver.implicitly_wait(10)


# 1. main page
path="http://apps.webofknowledge.com/UA_GeneralSearch_input.do?product=UA&SID=8BvIDyff6PjH9UimPl6&search_mode=GeneralSearch"
driver.get(path)  


# 2. checkbox
# checkboxes = driver.find_elements_by_xpath("//input[@type='checkbox'][@class='wos-style-checkbox']")
selected_checkboxes = [driver.find_element_by_id("WOS")]
not_selected_checkboxes = [ 
    driver.find_element_by_id("CSCD"),
    driver.find_element_by_id("DIIDW"),
    driver.find_element_by_id("KJD"),
    driver.find_element_by_id("MEDLINE"),
    driver.find_element_by_id("RSCI"),
    driver.find_element_by_id("SCIELO")]
if selected_checkboxes and not_selected_checkboxes:  # 判断是否有找到元素
    for checkbox in selected_checkboxes:  # 循环点击找到的元素
        r = checkbox.is_selected()
        if r:   # 如果被选中，结束这一次循环
            continue
        else:
            checkbox.click()  # 勾选复选框
        time.sleep(2)
    for checkbox in not_selected_checkboxes:  # 循环点击找到的元素
        r = checkbox.is_selected()
        if r:   # 如果被选中，取消选中
            checkbox.click()
        else:
            continue
        time.sleep(2)
else:
    print("没有找到元素")

time.sleep(5)

