from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import winsound as ws

driver = webdriver.Chrome("C:\\Users\\myeon\\Downloads\\chromedriver.exe")
driver.get("http://sugang.snu.ac.kr/sugang/cc/cc100.action")
search = driver.find_element_by_name('srchSbjtCd')
search.click()
search.send_keys('251.320')
search.send_keys(Keys.ENTER)

while 1:
    driver.find_element_by_css_selector('#cond00 > a.btn_search_ok').click()
    pop = driver.find_element_by_css_selector('#content > div > div.seach_cont.mt_30 > div.tbl_sec > div.gray_top > table > tbody > tr:nth-child(1) > td:nth-child(22)').get_property('innerText')
    if int(pop) < 80:
        for i in range(10):
            ws.Beep(4000,200)
            ws.Beep(3000,200)
    time.sleep(0.25)
