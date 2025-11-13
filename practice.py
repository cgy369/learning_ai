import random
import requests
import crawler

# from flask import Flask, jsonify, request
# import flask

# app = flask.Flask(__name__)


class Calculator:
    def __init__(self):
        self.result = 0  # result 초기화

    def operator(self, flag, a):
        if flag == "+":
            self.result += a
            return self.result
        elif flag == "-":
            self.result -= a
            return self.result
        else:
            if a == 0:
                return "Error: Division by zero"
            return self.result + a, self.result - a, self.result / a, self.result % a

    def refresh(self):
        self.result = 0


def main():
    calc1 = Calculator()
    print(calc1.operator("+", 3))
    print(calc1.operator("+", 3))
    print(calc1.operator("+", 3))
    calc1.refresh()
    print(calc1.operator("", 3))
    # num = random.randint(1, 5)
    # print(num)
    # print(4**2)
    # print(pow(4, 2))
    # print(crawler.calc("+", 4, 3))
    # res = crawler.calc("", 10, 2)
    # print(res[0])
    # response = requests.get("https://api.github.com")
    # print(response.status_code)
    # print(response.json())
    # result = crawler.login_quotes_toscrape("id", "pass")
    # print(result)


if __name__ == "__main__":
    main()
