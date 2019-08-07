import requests
import regex as re
from bs4 import BeautifulSoup as BS
import time
from selenium import webdriver
from urllib import parse
import os


# download pdf file
def download(url, sess, cookies, filename="test"):
    print("Downloading : ", filename)
    get_file_flag = True
    get_file_max_times = 5
    while get_file_flag:
        try:
            file = sess.get(url, cookies=cookies)
            get_file_flag = False
        except Exception as e:
            get_file_max_times -= 1
            if get_file_max_times == 0:
                print("Downloading : ", filename, " failed.")
                print("")
                print("")
                return
            print("Error: ", e)

    try:
        with open(filename + ".pdf", "wb") as f:
            f.write(file.content)
            print("Already downloaded in ", filename)
            print("")
            print("")
    except Exception as e:
        print("Error: ", e)


def search(sess, load_time, pay_load, cookie, root_path, resourses_path, local_path, chrome=None, page=1):
    page_dir = local_path + "/page_" + str(page)
    if not os.path.exists(page_dir):
        os.mkdir(page_dir)

    try_times = 5
    flag = False
    try_num = 0
    while try_times != 0:
        load_page_flag = True
        page_error_count = 5
        while load_page_flag:
            try:
                indexes = sess.post(resourses_path, data=pay_load, cookies=cookie)
                load_page_flag = False
                page_error_count -= 1
            except Exception as e:
                load_page_flag = True
                if page_error_count == 0:
                    return None
                print("Error:", e)

        connect_error = "connect()连接127.0.0.1:6600失败,错误号:10061."
        html_contents = indexes.text
        # print(html_contents)
        if html_contents.find(connect_error) == 1:
            try_num += 1
            try_times -= 1
            print("Can't connect server, try ", try_num, "time(s).")
            time.sleep(30)
        else:
            flag = True
            break

    if not flag:
        print("Wait some minutes, try again.")
        return None

    time.sleep(load_page_time)
    soup = BS(indexes.text, "html.parser")
    soup.prettify()

    numbers = None
    if page == 1:
        pdf_numbers_info = soup.find("div", class_="search_gs")
        pdf_numbers_pattern = "([0-9]*)篇;"
        pdf_numbers = re.findall(pdf_numbers_pattern, str(pdf_numbers_info))
        pdf_numbers = pdf_numbers[0]
        if pdf_numbers != '':
            numbers = int(pdf_numbers)
            print("Total find ", numbers, "files")
        else:
            return None

    # 3.parser current page
    pattern = "downpaper\('(.*)'\);return false"
    results = soup.find_all("span", class_="down")
    print("Download page ", page, ".")
    for i, raw_link in enumerate(results):
        a = raw_link.find_all("a")
        if len(a) == 1:
            print("============================================")
            print("Current file don't have a download link.")
            print("")
            print("")
            continue
        link = re.findall(pattern, str(a[1]))
        download_url_website = link[0].replace("&amp;", "&")
        url = root_path + download_url_website

        pattern_title = "(&T)=(.*?)$"
        pattern_title_obj = re.compile(pattern_title)
        raw_title = re.findall(pattern_title_obj, url)[0][1]
        filename = parse.unquote(raw_title)
        filename = filename.replace("/", "")
        print("============================================")
        print("Preparing downloading: ", filename)
        file_path = page_dir + "/" + filename + ".pdf"
        if os.path.exists(file_path):
            print(filename, "already in dir : ", page_dir)
            print("")
            print("")
            continue

        # time.sleep(5)
        download_not_finish_flag = True
        error_count = 5
        while download_not_finish_flag:
            try:
                chrome.get(url)
                error_count -= 1
                download_not_finish_flag = False
            except Exception as e:
                print("Error:", e)
                if error_count == 0:
                    continue

        download_not_finish_flag = True
        error_count = 5
        while download_not_finish_flag:
            try:
                chrome.get(url)
                error_count -= 1
                download_not_finish_flag = False
            except Exception as e:
                print("Error:", e)
                if error_count == 0:
                    continue

        time.sleep(load_time)
        hrefs = chrome.find_elements_by_xpath("//*[@href]")

        find_url_flag = False
        for href in hrefs:
            if href.text == '下载地址' or href.text == '镜像站-高速下载-1':
                find_url_flag = True
                download_url = href.get_attribute('href')
                print("Download url : ", download_url)
                download(download_url, sess, cookies={}, filename=page_dir + "/" + filename)
        if not find_url_flag :
            print("not find")
    if page == 1:
        return numbers


def begin():

    if not os.path.exists(keywords):
        os.mkdir(keywords)

    local_path = keywords

    with requests.Session() as sess:
        root = "http://119.1.159.163:81"

        # 1.first login
        login_pay_load = {"ecmsfrom": '', "enews": "login", 'username': username, 'password': password}
        login_flag = True
        log_times = 5
        while login_flag:
            try:
                response_login = requests.post(
                    url="http://www.797g.cn/e/member/doaction.php",
                    data=login_pay_load,
                    headers={
                        'user-agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
                    },
                )
                ck = response_login.cookies
                login_flag = False
            except Exception as e:
                print("Error: ", e)
                log_times -= 1
                print("Login failed, try again.")
                if log_times == 0:
                    print("Login error, wait some minutes and then try again.")
                    return None

        if ck:
            print("Login successful!")
        else:
            print("Login failed!")
        # 2.search
        resourses = root + "/ZK/search.aspx"
        pay_load = {
            "E":"(Keyword_C=(" + keywords +")+Title_C=(" + keywords + "))",
            "H": "题名或关键词=" + keywords + " 与 范围=" + scope + "",
            "M": "", "P": "1", "CP": "0", "CC": "0", "LC": "0",
            "Entry": "M", "S": "1", "SJ": "", "ZJ": "", "GC": "", "Type": ""
            }

        chrome = webdriver.Firefox()
        pdf_numbers = search(sess, load_time, pay_load, ck, root, resourses, local_path, chrome, page=1)
        if pdf_numbers is None:
            print("Can't find ", keywords, ", or wait some minutes, try again.")
        else:
            pages = pdf_numbers // 20
            if pdf_numbers % 20:
                pages += pages

            for i in range(2, pages + 1):
                pay_load['P'] = str(i)
                search(sess, load_time, pay_load, ck, root, resourses, local_path, chrome, page=i)
        chrome.close()


if __name__ == "__main__":
    username = "58574628996199"  # 用户名
    password = "437965"  # 密码
    keywords = "分布式数据库"  # 关键字
    scope = "全部期刊"  # 搜索范围,可选字段
    load_page_time = 30
    load_time = 20
    for i in range(5):
        load_time += 10
        load_page_time += 10
        begin()

