import requests
from bs4 import BeautifulSoup


def login_quotes_toscrape(username, password):
    login_url = "https://quotes.toscrape.com/login"
    session = requests.Session()

    # 로그인 페이지에서 CSRF 토큰 가져오기
    resp = session.get(login_url)
    soup = BeautifulSoup(resp.text, "html.parser")
    csrf_token = soup.find("input", {"name": "csrf_token"})["value"]

    login_data = {"csrf_token": csrf_token, "username": username, "password": password}

    post_resp = session.post(login_url, data=login_data)

    if post_resp.ok:
        print("로그인 성공")
        # 로그인 후 첫 페이지 가져오기
        profile_page = session.get("https://quotes.toscrape.com/")
        return profile_page.text
    else:
        print("로그인 실패")
        return None


def calc(flag, a, b):
    if flag == "+":
        return a + b
    elif flag == "-":
        return a - b
    else:
        return a + b, a - b, a / b, a % b


if __name__ == "__main__":
    print(calc("+", 1, 2))
