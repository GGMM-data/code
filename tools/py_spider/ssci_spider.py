# !/usr/bin/env python
# -*- coding:utf-8 -*-
from selenium import webdriver
import time
from tqdm import tqdm
from multiprocessing import Pool


def get_driver():
	options = webdriver.ChromeOptions()
	prefs = {
		'profile.default_content_setting_values':
			{
				'notifications': 2
			}
	}
	options.add_experimental_option('prefs', prefs)  # 关掉浏览器左上角的通知提示
	options.add_argument("disable-infobars")  # 关闭'chrome正受到自动测试软件的控制'提示
	driver = webdriver.Chrome(options=options)
	driver.maximize_window()
	driver.implicitly_wait(20)
	return driver


def download(theme):
	driver = get_driver()
	# 1. 获取检索主界面
	path = "http://apps.webofknowledge.com/WOS_GeneralSearch_input.do?product=WOS&SID=8BvIDyff6PjH9UimPl6&search_mode=GeneralSearch"
	# 获取检索页面
	driver.get(path)
	
	# 2. 选择检索范围
	selected_checkboxes = [
		driver.find_element_by_id("editionitemSSCI")]
	not_selected_checkboxes = [
		driver.find_element_by_id("editionitemSCI"),
		driver.find_element_by_id("editionitemISTP"),
		driver.find_element_by_id("editionitemISSHP"),
		driver.find_element_by_id("editionitemBSCI"),
		driver.find_element_by_id("editionitemBHCI"),
		driver.find_element_by_id("editionitemCCR"),
		driver.find_element_by_id("editionitemIC")]
	if selected_checkboxes and not_selected_checkboxes:  # 判断是否有找到元素
		for checkbox in selected_checkboxes:  # 循环点击找到的元素
			r = checkbox.is_selected()
			if r:  # 如果被选中，结束这一次循环
				continue
			else:
				checkbox.click()  # 勾选复选框
			time.sleep(0.2)
		for checkbox in not_selected_checkboxes:  # 循环点击找到的元素
			r = checkbox.is_selected()
			if r:  # 如果被选中，取消选中
				checkbox.click()
			else:
				continue
			time.sleep(0.2)
	else:
		print("没有找到元素")
	
	# 3.输入检索内容
	keywords = driver.find_element_by_id("value(input1)")
	# 清空表单
	keywords.clear()
	# 提交表单
	keywords.send_keys(theme)
	# 4.搜索
	searchbutton = driver.find_element_by_id("searchCell1")
	searchbutton.click()
	# 5.获取搜索结果总页数
	pagecount = driver.find_element_by_id("pageCount.bottom").text
	pagecount = pagecount.replace(",", "")
	#print(theme, pagecount)
	
	# 6.记录每一页搜索结果
	print(theme)
	f = open(theme, "a+")
	for page in tqdm(range(1, int(pagecount))):
		print(page)
		page_button = driver.find_element_by_xpath("//input[@type='text'][@name='page']")
		page_button.clear()
		page_button.send_keys(str(page)+"\n")

		results = driver.find_elements_by_xpath("//value[@lang_id]")
		for r in results:
			info = r.text
			f.write(info+"\n")
	f.close()
	driver.close()


if __name__ == "__main__":
	#themes = ["agro*", "fertilizer", "pesticide", "crop", "poverty, Hunger",
	#	"urbanization", "urban-rural", "township enterprises", "Agri*"]
	themes = ["peasant", "rural", "land", "food", "farmer"]

	for t in themes:
		download(t)
	# cpu_numbers = 10
	# pool = Pool(processes=cpu_numbers)
	# pool.map(download, themes)