import ssl
from openpyxl import Workbook
import requests
from bs4 import BeautifulSoup
import re
from time import sleep

ssl._create_default_https_context = ssl._create_unverified_context

# Request Header
headers = {
    'Connection': 'keep-alive',
    'Accept-Encoding': 'gzip, deflate, br',
    'Cache-Control': 'no-cache',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36',
}

ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')

# 创建一个workbook 设置编码
workbook = Workbook()

# 创建一个worksheet
worksheet = workbook.active


def request(url):
    try:
        r = requests.get(url=url, headers=headers, timeout=20)

        bs = BeautifulSoup(r.text, "html.parser")

        table = bs.select('.container')[0]

        return table
    except Exception as e:
        print('访问url: ' + url + ' error: ' + str(e))


def write_excel(link, title, vote_count, view_count, answer_count, question, answers):
    worksheet.append([link, title, vote_count, view_count, answer_count, question, answers])


def get_post_content(tags):
    res = ''

    if tags.contents is None:
        return ''

    for tag in tags.contents:
        if type(tag).__name__ == 'Tag':
            if tag.name == 'p' or tag.name == 'pre':
                res += tag.get_text() + '\r\n'
            elif tag.contents is not None:
                res += get_post_content(tag)
    return res


def get_post_comment(tag):
    comments = ''
    c_c = 1
    for c in tag.select('.comment-copy'):
        comments += 'Comment' + str(c_c) + '\r\n' + c.get_text() + '\r\n'
        c_c += 1

    return comments


def save_question_info(question_url):
    table = request(question_url)

    question_related = table.select('#question')[0]

    post_text = question_related.select('.post-text')[0]

    # 获取question 内容及评论
    question_content = get_post_content(post_text)
    question_comment = get_post_comment(question_related)

    question = '\r\n' + question_content + '\r\n' + 'Comments for this question' + '\r\n' + question_comment + '\n'

    # 获取answers
    answer_related = table.select('#answers')[0]
    answers = ''
    a_c = 1

    for a in answer_related.contents:
        if type(a).__name__ == 'Tag':
            if a.name == 'div' and (a.get('class') is not None and 'answer' in a.get('class')):
                answer_content = get_post_content(a.select('.post-text')[0])
                answer_comment = get_post_comment(a)

                answers += 'Answer' + str(
                    a_c) + '\r\n' + answer_content + '\r\n' + 'Comments for this answer:' + '\r\n' + answer_comment + '\n '

                a_c += 1

    return question, answers


def save_info(page, table):
    # 查找检索结果项
    search_res = table.select('#questions')[0]

    # 记录所有的question id
    question_lists = search_res.select('.question-summary')

    cur = 0
    success = 0

    for q in question_lists:
        cur += 1
        q_id = -1

        try:
            # 获得id
            q_id = str(q.get('id')).split('-')[2]

            # 获得vote, answer以及view数量
            tmp = q.find_all('strong')
            vote_count = tmp[0].get_text()
            answer_count = tmp[1].get_text()

            view_count = str(q.select('.views')[0].get('title')).replace(' views', '')

            # 记录标题
            title = q.select('.question-hyperlink')[0].get_text()

            # 链接
            link = "https://stackoverflow.com/questions/" + q_id

            # 获取该问题的详细信息
            question, answers = save_question_info(link)

            # 写入表格中
            write_excel(link, title, vote_count, view_count, answer_count, question, answers)

            # 计数加一
            success += 1

            sleep(5)

        except Exception as e:
            print('爬取错误 page: ' + str(page) + ' item: ' + str(cur) + ' question id: ' + str(q_id) + ' error: ' + str(e))
            continue


if __name__ == '__main__':

    total_pages = 503

    # Search Url
    url = 'https://stackoverflow.com/questions/tagged/architecture?tab=newest&page='
    url_append = '&pagesize=30'

    for page in range(1, total_pages + 1):
        try:
            print('正在爬取页面: ' + str(page))
            search_url = url + str(page) + url_append

            t = request(search_url)
            save_info(page, t)
        except Exception:
            continue

    workbook.save('C:\\Users\\Adminstrator\\Desktop\\results.xls')
    print('OK')
    
    
    
